# -*- coding: utf-8 -*-
'''
DQN architecture
Folk from: https://github.com/keon/deep-q-learning
'''
from __future__ import print_function
import os
import sys
import random
import numpy as np
from collections import deque
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, concatenate, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from attention_block import SoftAttentionBlock
from itertools import groupby
from operator import itemgetter

EPISODES = 8000
segLen=48
track_num=2
pianoKeys=60
vecLen=pianoKeys*track_num
maxdelta=33
hidden_delta=128
hidden_note=256
drop_rate=0.2

class PGAgent:
    def __init__(self, lr=1e-7, gamma=0.95): ## low lr to tune all weights
        self.learning_rate = lr
        self.model = self._build_model()
        self.notes = [] # 1
        self.deltas= [] # 2
        self.pnotes= [] # 3
        self.pdeltas=[] # 4
        self.notes_grad  =[] #5
        self.deltas_grad =[] #6
        self.notes_reward=[] #7
        self.deltas_reward=[] #8
        self.gamma = gamma
        self.randomT = random.SystemRandom()
    def weighted_choice(self, choices):
        total = sum(choices)
        r = self.randomT.uniform(0, total)
        upto = 0
        for i, w in enumerate(choices):
            if upto + w >= r:
                return i
            upto += w
        return np.random.choice(len(choices), 1, p=choices)[0]
    def reset(self):
        self.notes, self.deltas, self.pnotes, self.pdeltas, self.notes_grad, self.deltas_grad, self.notes_reward, self.deltas_reward = [], [], [], [], [], [], [], []

    def _build_model(self):
        # Neural Net for PG learning Model
        noteInput  = Input(shape=(segLen, vecLen))
        noteEncode = LSTM(hidden_note, return_sequences=True, dropout=drop_rate)(noteInput)
        noteEncode = LSTM(hidden_note, return_sequences=True, dropout=drop_rate)(noteEncode)

        deltaInput = Input(shape=(segLen, maxdelta))
        deltaEncode = LSTM(hidden_delta, return_sequences=True, dropout=drop_rate)(deltaInput)
        deltaEncode = LSTM(hidden_delta, return_sequences=True, dropout=drop_rate)(deltaEncode)

        codec = concatenate([noteEncode, deltaEncode], axis=-1)
        codec = SoftAttentionBlock(codec, segLen, hidden_note+hidden_delta)
        codec = LSTM(600, return_sequences=True, dropout=drop_rate, activation='softsign')(codec)
        codec = LSTM(600, return_sequences=False, dropout=drop_rate, activation='softsign')(codec)
        encoded = Dropout(drop_rate)(codec)

        fc_notes = BatchNormalization()(encoded)
        pred_notes = Dense(vecLen, kernel_initializer='normal', activation='softmax', name='note_output')(fc_notes) ## output score

        fc_delta = BatchNormalization()(encoded)
        pred_delta = Dense(maxdelta, kernel_initializer='normal', activation='softmax', name='time_output')(fc_delta) ## output score
        model = Model([noteInput, deltaInput], [pred_notes, pred_delta])

        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=self.learning_rate, clipnorm=1., decay=1e-7, momentum=0.9, nesterov=True))
        return model

    def remember(self, note_act, delta_act, state_note, state_delta, reward_note, reward_delta, prob_note, prob_delta):
        ## appends all:
        self.notes.append(state_note)
        self.deltas.append(state_delta)
        self.notes_reward.append(reward_note)
        self.deltas_reward.append(reward_delta)
        self.pnotes.append(prob_note)
        self.pdeltas.append(prob_delta)
        new_pn = np.zeros(vecLen, dtype=np.float32)
        new_pn[note_act] = 1.
        self.notes_grad.append(new_pn - prob_note)
        new_pd = np.zeros(maxdelta, dtype=np.float32)
        new_pd[delta_act] = 1.
        self.deltas_grad.append(new_pd - prob_delta)

    def act(self, state): ## Using CPU
        act_note, act_delta = self.model.predict(state)
        #act_note = act_note[0] / np.sum(act_note[0]) ## norm -> PMF
        #act_delta = act_delta[0] / np.sum(act_delta[0])
        #return np.random.choice(vecLen, 1, p=act_note)[0], np.random.choice(maxdelta, 1, p=act_delta)[0], act_note, act_delta
        return self.weighted_choice(act_note[0]), self.weighted_choice(act_delta[0]), act_note[0], act_delta[0]
    def _discount_and_norm_rewards(self, reward):
        '''
        See: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py
        , http://karpathy.github.io/2016/05/31/rl/ and
        https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/5-2-policy-gradient-softmax2/
        for further explain

        '''
        discount_rs = np.zeros_like(reward)
        running_add = 0
        for t in reversed(range(len(reward))):
            running_add = running_add * self.gamma + reward[t]
            discount_rs[t] = running_add
        # normalize the rewards
        discount_rs -= discount_rs.mean()
        discount_rs /= discount_rs.std()
        return discount_rs

    def train(self): ## Using GPU
        grad_n = np.vstack(self.notes_grad)
        grad_d = np.vstack(self.deltas_grad)
        reward_n = np.vstack(self.notes_reward)
        reward_d = np.vstack(self.deltas_reward)
        grad_n *= self._discount_and_norm_rewards(reward_n)
        grad_d *= self._discount_and_norm_rewards(reward_d)
        notes = np.array(self.notes)
        deltas= np.array(self.deltas)
        target_n = np.vstack(self.pnotes) + self.learning_rate * grad_n
        target_d = np.vstack(self.pdeltas) + self.learning_rate * grad_d
        self.model.train_on_batch([notes[:,0,:,:], deltas[:,0,:,:]], [target_n, target_d])
        self.reset() ## forget it

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def lrs(text):
    import numpy as np
    _, _, lcp = suffix_array(text)
    return np.max(np.array(lcp))

def suffix_array(text, _step=16):
    tx = text
    size = len(tx)
    step = min(max(_step, 1), len(tx))
    sa = list(range(len(tx)))
    sa.sort(key=lambda i: tx[i:i + step])
    grpstart = size * [False] + [True]  # a boolean map for iteration speedup.
    # It helps to skip yet resolved values. The last value True is a sentinel.
    rsa = size * [None]
    stgrp, igrp = '', 0
    for i, pos in enumerate(sa):
        st = tx[pos:pos + step]
        if st != stgrp:
            grpstart[igrp] = (igrp < i - 1)
            stgrp = st
            igrp = i
        rsa[pos] = igrp
        sa[i] = pos
    grpstart[igrp] = (igrp < size - 1 or size == 0)
    while grpstart.index(True) < size:
        # assert step <= size
        nextgr = grpstart.index(True)
        while nextgr < size:
            igrp = nextgr
            nextgr = grpstart.index(True, igrp + 1)
            glist = []
            for ig in range(igrp, nextgr):
                pos = sa[ig]
                if rsa[pos] != igrp:
                    break
                newgr = rsa[pos + step] if pos + step < size else -1
                glist.append((newgr, pos))
            glist.sort()
            for ig, g in groupby(glist, key=itemgetter(0)):
                g = [x[1] for x in g]
                sa[igrp:igrp + len(g)] = g
                grpstart[igrp] = (len(g) > 1)
                for pos in g:
                    rsa[pos] = igrp
                igrp += len(g)
        step *= 2
    del grpstart
    # create LCP array
    lcp = size * [None]
    h = 0
    for i in range(size):
        if rsa[i] > 0:
            j = sa[rsa[i] - 1]
            while i != size - h and j != size - h and tx[i + h] == tx[j + h]:
                h += 1
            lcp[rsa[i]] = h
            if h > 0:
                h -= 1
    if size > 0:
        lcp[0] = 0
    return sa, rsa, lcp

class rewardSystem:
    def __init__(self, rat, oldr, model_dir=None):
        self.rewardRNN = None
        if not model_dir is None:
            self.rewardRNN = [ (load_model(str(model_dir)+'/'+r), self.fn2float(r)) for r in os.listdir(str(model_dir)) ]
        self.state_note = np.zeros((1, segLen, vecLen), dtype=np.bool)
        self.state_delta= np.zeros((1, segLen, maxdelta), dtype=np.bool)
        self.firstNote = None
        self.c = rat
        self.d = oldr
    def fn2float(self, s):
        return float('.'.join(s.split('_')[-1].split('.')[:-1]))
    def reset(self, seed=None):
        if seed is None: ## if seed is not specified, sets to 0
            self.state_note[:,:,:] = 0
            self.state_delta[:,:,:]= 0
        else: # using seed
            seed = np.load(str(seed))
            seedIdx = np.random.randint(len(seed['notes']))
            self.state_note[:,:,:] = seed['notes'][seedIdx,:,:]
            self.state_delta[:,:,:]= seed['times'][seedIdx,:,:]
        self.firstNote = None
    def countFinger(self, x, y, deltas, notes):
        if x>0: return 1
        cnt=1 ## self
        for i, v in enumerate(reversed(deltas)): ## others
            cnt += 1 if self.sameTrack(y, notes[segLen-1-i]) else 0
            if v!=0: break
        return cnt
    def countSameNote(self, x, l):
        cnt = 1
        for v in reversed(l):
            if self.sameTrack(v, x):
                if v==x: cnt+=1
                else: break
        return cnt
    def get_state(self):
        return self.state_note, self.state_delta
    def scale(self, diffLastNote, delta):
        if diffLastNote>12: ## Too big jump
            return -1
        elif diffLastNote==4 or diffLastNote==7: ## western
            return 1
        elif diffLastNote<=2 and delta==0: ## annoying sound
            return -1
        elif diffLastNote==12 and delta==0: ## full 8
            return 1
        return 0
    def sameTrack(self, a, b):
        return (a<pianoKeys and b<pianoKeys) or (a>=pianoKeys and b>=pianoKeys)
    def checkTrackDist(self, note, delta, notes, deltas):
        accumTick=delta
        i = None
        for ti, v in enumerate(reversed(deltas)):
            i = segLen-1-ti
            if not self.sameTrack(note ,notes[i]): break ## find accompany
            i = None
            accumTick += v
        return accumTick, i
    def findRootNote(self, idx, notes, deltas):
        rootN = notes[idx]-pianoKeys ## find root note of accompany
        for i in reversed(range(idx)): ## reverse([0,idx))
            if notes[i]>=pianoKeys: ## is accompany
                rootN = min(rootN, notes[i]-pianoKeys) ## at smaller pitch
                if deltas[i]>0: break ## if not stack on others, break
            else: break ## not an accompany note
        return rootN
    def reward(self, action_note, action_delta, verbose=False):
        done = False
        pitchStyleReward = 0.
        tickStyleReward = 0.
        tot_r = 0.
        if not self.rewardRNN is None:
            for i, (m, r) in enumerate(self.rewardRNN):
                p_n, p_d = m.predict([self.state_note, self.state_delta], verbose=0)
                pitchStyleReward += math.log(p_n[0][action_note])*r
                tickStyleReward += math.log(p_d[0][action_delta])*r
                tot_r += r
            pitchStyleReward /= tot_r
            tickStyleReward /= tot_r
        reward_note=0
        reward_delta= -1 if action_delta>0 and action_delta<4 else 0 ## prevent unfavored tick: (0,4)
        if np.sum(self.state_note)==segLen and np.sum(self.state_delta)==segLen:
            state_idx_note = [ np.where(r==1)[0][0] for r in self.state_note[0] ]
            state_idx_delta = [ np.where(r==1)[0][0] for r in self.state_delta[0] ]
            if self.countSameNote(action_note, state_idx_note)>4:
                done = True ## bad end
            dist, idx = self.checkTrackDist(action_note, action_delta, state_idx_note, state_idx_delta)
            if action_note<pianoKeys: ## is main melody
                if not idx is None: ## idx points to a nearest accompany note
                    ## find root note
                    rootN = self.findRootNote(idx, state_idx_note, state_idx_delta)
                    dist = abs(rootN-action_note)%12
                    if dist==0 or dist==7 or dist==4: ## check if valid chord
                        reward_note += 1
            ## scale score, not complete yet...
            idx = None
            for i, v in enumerate(reversed(state_idx_note)):
                if self.sameTrack(v, action_note):
                    idx = -1-i ## find last note in current track
                    break
                elif state_idx_delta[-1-i]>0:
                    break
            if not idx is None:
                diffLastNote = abs(action_note - state_idx_note[idx])
                reward_note += self.scale(diffLastNote, action_delta)
            ## check if generate longer longest repeat substring
            lrsi = state_idx_note
            lrsNote_old = lrs(lrsi)
            lrsi.append(action_note)
            lrsNote_new = lrs(lrsi)
            diff = lrsNote_new - lrsNote_old
            if lrsNote_new == 0:
                reward_note -= 1
                done = True
            elif diff>0: ## check update
                if lrsNote_new>8: ## ??
                    reward_note += 1
                if lrsNote_new>12:
                    reward_note -= 1
                    done = True ## bad end, very bad...
            ## not complete yet...
            if action_note<pianoKeys: ## main
                reward_delta -= 0 if self.countFinger(action_delta, action_note, state_idx_delta, state_idx_note) <= 5 else 1
            else: ## accompany
                reward_delta -= 0 if self.countFinger(action_delta, action_note, state_idx_delta, state_idx_note) <= 4 else 1

        ## update state:
        self.state_note = np.roll(self.state_note, -1, axis=1)
        self.state_note[0,-1,:] = 0
        self.state_note[0,-1,action_note] = 1

        self.state_delta = np.roll(self.state_delta, -1, axis=1)
        self.state_delta[0,-1,:] = 0
        self.state_delta[0,-1,action_delta] = 1
        if self.firstNote is None:
            self.firstNote = action_note
        if verbose:
            sys.stderr.write("reward_note = %d, %.2f, %s\n" % (reward_note, pitchStyleReward, "T" if done else "F"))
            sys.stderr.write("reward_delta = %d, %.2f\n" % (reward_delta, tickStyleReward))
        reward_note = reward_note*self.c+self.d*pitchStyleReward
        reward_delta= reward_delta*self.c+self.d*tickStyleReward
        return reward_note, reward_delta, done


if __name__ == "__main__":
    agent = PGAgent(lr=1e-7, gamma=0.99)
    agent.load(str(sys.argv[1]))
    seedPos = str(sys.argv[2])
    rewardSys = rewardSystem(0.2,1,model_dir = str(sys.argv[3])) ## more sensitive
    done = False
    batch_size = 128

    with open('./pg.csv', 'a+', 0) as logFP: ## no-buffer logging
        logFP.write('pitch, tick\n')
        rewardSys.reset(seed=seedPos) ## initialize states
        score_note = 0.
        score_delta = 0.
        for e in xrange(EPISODES):
            snote, sdelta = rewardSys.get_state() ## give initial state
            for time in xrange(batch_size):
                action_note, action_delta, p_n, p_d = agent.act([snote, sdelta]) ## action on state
                reward_note, reward_delta, done = rewardSys.reward(action_note, action_delta, verbose=False) ## reward on state
                score_note += float(reward_note)
                score_delta += float(reward_delta)
                nnote, ndelta = rewardSys.get_state() ## get next state
                agent.remember(action_note, action_delta, snote, sdelta, float(reward_note), float(reward_delta), p_n, p_d)
                snote, sdelta = nnote, ndelta ## update current state
                if done: ## termination
                    rewardSys.reset(seed=seedPos) ## new initial state
                    break
            sys.stderr.write('episode: %d Learning from past... bs: %d\n' % (e, len(agent.notes)))
            logFP.write("%.2f, %.2f\n" % (score_note, score_delta))
            score_note, score_delta = 0, 0
            agent.train()
            if e % 10 == 0:
                agent.save("./pg/melody-ddqn-{}.h5".format(e))
