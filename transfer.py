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
from keras.layers import Dense, Input, concatenate, Dropout, Activation, LSTM
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
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
drop_rate=0.3

class PGAgent:
    def __init__(self, lr=1e-7, gamma=0.95, batch_size=128): ## low lr to tune all weights
        self.batch_size = batch_size
        self.learning_rate = lr
        self.model = None
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
        self.model.fit([notes[:,0,:,:], deltas[:,0,:,:]], [target_n, target_d], batch_size=self.batch_size, epochs=1)
        self.reset() ## forget it

    def load(self, name):
        self.model = load_model(name)

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

def acf(x, l):
    x = np.array(x)
    y1 = x[:-l]
    y2 = x[l:]
    xm = x.mean()
    sump = np.sum((y1-xm)*(y2-xm))
    return sump / (x.var()*(len(x)-l))

class rewardSystem:
    def __init__(self, model_r, mt_r, model_dir=None): ## higher rat -> more mt score
        self.rewardRNN = None
        if not model_dir is None:
            self.rewardRNN = [ (load_model(str(model_dir)+'/'+r), self.fn2float(r)) for r in os.listdir(str(model_dir)) ]
        self.state_note = np.zeros((1, segLen, vecLen), dtype=np.bool)
        self.state_delta= np.zeros((1, segLen, maxdelta), dtype=np.bool)
        self.firstNote = None
        self.d = model_r ## score from original model
        self.c = mt_r ## music theorem score rate
        self.tick_counter = 0
        self.actions_note = []
        self.actions_delta = []
    def fn2float(self, s):
        return float('.'.join(s.split('_')[-1].split('.')[:-1]))
    def reset(self):
        self.state_note[:,:,:] = 0
        self.state_delta[:,:,:]= 0
        self.firstNote = None
        self.actions_note = []
        self.actions_delta= []
        self.LA = []
        self.tick_counter = 0
    def countFinger(self, x, y, deltas, notes):
        if x>0: return 1
        cnt=1 ## self
        for i, v in enumerate(reversed(deltas)): ## others
            cnt += 1 if self.sameTrack(y, notes[len(deltas)-1-i]) else 0
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
        elif diffLastNote<=2 and delta==0: ## annoying sound
            return -1
        return 0
    def sameTrack(self, a, b):
        return (a<pianoKeys and b<pianoKeys) or (a>=pianoKeys and b>=pianoKeys)
    def checkTrackDist(self, note, delta, notes, deltas):
        accumTick=delta
        i = None
        for ti, v in enumerate(reversed(deltas)):
            i = len(deltas)-1-ti
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
    def checkTune(self):
        AC = lambda x, s: x in s
        tkeys = []
        for n in self.actions_note:
            if n<pianoKeys: ## main
                tkeys.append(int(n+36))
            else:
                tkeys.append(int(n-pianoKeys+36))
        noteHist, _ = np.histogram(np.array(tkeys)%12, bins=12)
        histArg = np.argsort(-noteHist)
        tune = set(histArg[:3]) ## decreaseing order, top 3
        key =  set(histArg[:7]) ## decreaseing order, top 7
        main = [ (v+36)%12 for i, v in enumerate(self.actions_note) if v<pianoKeys and self.actions_delta[i]>0 ]
        accompany = [ (v-pianoKeys+36)%12 for i, v in enumerate(self.actions_note) if v>=pianoKeys and self.actions_delta[i]>0 ]
        main_score = 0
        accompany_score = 0
        if len(main)>0:
            for n in main:
                if not n in key: main_score -= 1
            for i in xrange(1, len(main)):
                last = main[i-1]
                curr = main[i]
                lastA= AC(last,main)
                currA= AC(curr,main)
                if not lastA and currA: ## inact -> act
                    main_score += 1
            if main[0]==main[-1] and main[0]==histArg[0]: main_score += 1
            main_score /= float(len(main))
        if len(accompany)>0:
            for n in accompany:
                if not n in key: accompany_score -= 1
            for i in xrange(1, len(accompany)):
                last = accompany[i-1]
                curr = accompany[i]
                lastA= AC(last,accompany)
                currA= AC(curr,accompany)
                if not lastA and currA: ## inact -> act
                    accompany_score += 1
            accompany_score /= float(len(accompany))
        return main_score+accompany_score

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
        reward_delta=0
        idx = None
        if len(self.actions_note)>0:
            if self.sameTrack(action_note, self.actions_note[-1]):
                reward_note += self.scale(abs(action_note-self.actions_note[-1]), action_delta)
            if self.countSameNote(action_note, self.actions_note)>4: reward_note-=1
            dist, idx = self.checkTrackDist(action_note, action_delta, self.actions_note, self.actions_delta)
            if idx is None: ## idx points to a nearest accompany note
                reward_note -= 1 ## the other track is dead
        if len(self.actions_note)>0 and self.tick_counter%32+action_delta>=32: ## complete a half of segment
            if idx is None and np.sum(np.array(self.actions_delta)<=2)==len(self.actions_delta): ## too fast, too annoying
                reward_delta -= 1
            if action_note<pianoKeys: ## main
                reward_delta -= 0 if self.countFinger(action_delta, action_note, self.actions_delta, self.actions_note) <= 5 else 1
            else: ## accompany
                reward_delta -= 0 if self.countFinger(action_delta, action_note, self.actions_delta, self.actions_note) <= 5 else 1
            ## too many fingers
            reward_note += self.checkTune() ## score on melody
            self.LA.extend(self.actions_note) ## append one full segment into LA
            self.actions_note = []
            self.actions_delta= []
        if len(self.LA)>0 and self.tick_counter%512+action_delta>=512: ## 32*2*8 = 512, 8 full segments, needs large VRAMs
            done = True
            mean_acf = 0.
            acfl = min(len(self.LA),3)
            for l in xrange(acfl): mean_acf += acf(self.LA, l+1)
            mean_acf /= float(acfl)
            reward_note -= mean_acf ## penlty of acf
            lrsLA = lrs(self.LA)
            if lrsLA>=20: ## long phrase
                lrsnorm = lrsLA / float(len(self.LA))
                reward_note += lrsnorm ## longest repeated substring: longer string -> bigger structure
            self.LA = []

        self.tick_counter += action_delta
        self.actions_note.append(action_note)
        self.actions_delta.append(action_delta)

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
            sys.stderr.write("reward_note = %.2f, %.2f, %s\n" % (reward_note, pitchStyleReward, "T" if done else "F"))
            sys.stderr.write("reward_delta = %.2f, %.2f\n" % (reward_delta, tickStyleReward))
        return np.clip(reward_note,-1,1)*self.c+self.d*np.clip(pitchStyleReward,-1,1), np.clip(reward_delta,-1,1)*self.c+self.d*np.clip(tickStyleReward,-1,1), done, reward_note, reward_delta

if __name__ == "__main__":
    agent = PGAgent(lr=1e-7, gamma=0.99, batch_size=128)
    agent.load(str(sys.argv[1]))
    rewardSys = rewardSystem(0.2, 0.8,model_dir = str(sys.argv[2])) ## more sensitive, 0.2, 0.8
    done = False

    with open('./pg.csv', 'a+', 0) as logFP: ## no-buffer logging
        logFP.write('pitch, tick\n')
        rewardSys.reset() ## initialize states
        score_note = 0.
        score_delta = 0.
        for e in xrange(EPISODES):
            snote, sdelta = rewardSys.get_state() ## give initial state
            done = False
            while not done:
                action_note, action_delta, p_n, p_d = agent.act([snote, sdelta]) ## action on state
                reward_note, reward_delta, done, mt_note, mt_delta = rewardSys.reward(action_note, action_delta, verbose=True) ## reward on state
                score_note += float(mt_note)
                score_delta += float(mt_delta)
                nnote, ndelta = rewardSys.get_state() ## get next state
                agent.remember(action_note, action_delta, snote, sdelta, reward_note, reward_delta, p_n, p_d)
                snote, sdelta = nnote, ndelta ## update current state
            if len(agent.notes)<2:
                agent.reset()
                continue
            sys.stderr.write('episode: %d Learning from past... bs: %d\n' % (e, len(agent.notes)))
            logFP.write("%.2f, %.2f\n" % (score_note, score_delta))
            score_note, score_delta = 0, 0
            agent.train()
            if e % 20 == 0:
                agent.save("./pg/melody-ddqn-{}.h5".format(e))
                rewardSys.reset() ## new initial state
