# -*- coding: utf-8 -*-
'''
DQN architecture
Folk from: https://github.com/keon/deep-q-learning
'''
from __future__ import print_function
import os
os.environ['KERAS_BACKEND']='tensorflow'
import sys
import random
import numpy as np
from collections import deque
import math
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
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

def softmaxSample(a, temp=0.7):
    '''
    ref: https://github.com/itaicaspi/keras-dqn-doom, and https://stackoverflow.com/questions/34968722/softmax-function-python
    '''
    if a.ndim != 1: raise ValueError('softmax: Only support 1-D array!')
    try:
        e_a = np.exp((a-max(0.0, np.max(a))) / temp) ## max trick
        preds = e_a / e_a.sum()
        return np.argmax(np.random.multinomial(1, preds, 1))
    except:
        return np.argmax(a)

class DQNAgent:
    def __init__(self, policy='softmax', verbose=False):
        self.memory = deque(maxlen=16384)
        self.gamma = 0.8    # discount rate
        self.epsilon = 0.99  # exploration rate
        self.epsilon_min = 0.001 ## large eps
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.policy=policy
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        if verbose:
            if self.policy=='softmax':
                sys.stderr.write('Agent: Softmax policy\n')
            else:
                sys.stderr.write('Agent: E-greedy policy\n')

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
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
        pred_notes = Dense(vecLen, kernel_initializer='normal', activation='linear', name='note_output')(fc_notes) ## output score

        fc_delta = BatchNormalization()(encoded)
        pred_delta = Dense(maxdelta, kernel_initializer='normal', activation='linear', name='time_output')(fc_delta) ## output score
        model = Model([noteInput, deltaInput], [pred_notes, pred_delta])

        model.compile(loss=self._huber_loss,
                      optimizer=RMSprop(lr=self.learning_rate, clipnorm=1.))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state_note, state_delta, action_note, action_delta, reward_note, reward_delta,  next_state_note, next_state_delta, done):
        self.memory.append((state_note, state_delta, action_note, action_delta, reward_note, reward_delta,  next_state_note, next_state_delta, done))

    def act(self, state): ## Using CPU
        with tf.device('/cpu:0'):
            if self.policy=='softmax': ## softmax policy
                act_note, act_delta = self.model.predict(state)
                return softmaxSample(act_note[0]), softmaxSample(act_delta[0])
            else: ## epsilon-greedy
                if np.random.rand() <= self.epsilon:
                    return random.randrange(vecLen), random.randrange(maxdelta)
                act_note, act_delta = self.model.predict(state)
                return np.argmax(act_note[0]), np.argmax(act_delta[0])  # returns action

    def replay(self, batch_size): ## Using GPU
        with tf.device('/gpu:0'):
            minibatch = random.sample(self.memory, batch_size)
            batch_note = np.zeros((batch_size, segLen, vecLen), dtype=np.bool)
            batch_delta= np.zeros((batch_size, segLen, maxdelta), dtype=np.bool)
            #state_note(0), state_delta(1), action_note(2), action_delta(3), reward_note(4), reward_delta(5),  next_state_note(6), next_state_delta(7), done(8) = entries
            for i, entries in enumerate(minibatch):
                batch_note[i,:,:] = entries[0][0] ## state_note
                batch_delta[i,:,:]= entries[1][0] ## state_delta
            state_notes = batch_note ## a batch of state_note
            state_deltas= batch_delta
            target_notes, target_deltas = self.model.predict([batch_note, batch_delta]) ## get a batch of target
            for i, entries in enumerate(minibatch):
                batch_note[i,:,:] = entries[6][0] ## next_state_note
                batch_delta[i,:,:]= entries[7][0] ## next_state_delta
            a_notes, a_deltas = self.model.predict([batch_note, batch_delta]) ## get a batch of q-value of new model
            t_notes, t_deltas = self.target_model.predict([batch_note, batch_delta]) ## old model
            for i, entries in enumerate(minibatch):
                target_notes[i][entries[2]] = entries[4] + (self.gamma*t_notes[i][np.argmax(a_notes[i])] if not entries[8] else 0) ## target_note()(act_note) = rew_note
                target_deltas[i][entries[3]] = entries[5]+ (self.gamma*t_deltas[i][np.argmax(a_deltas[i])] if not entries[8] else 0) ## note -> delta ''
            self.model.fit([state_notes, state_deltas], [target_notes, target_deltas], epochs=1, verbose=0) ## a minibatch

    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return self.epsilon

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
    def __init__(self, rat, oldr):
        self.rewardRNN = load_model("./NoteRNN.h5")
        self.state_note = np.zeros((1, segLen, vecLen), dtype=np.bool)
        self.state_delta= np.zeros((1, segLen, maxdelta), dtype=np.bool)
        self.firstNote = None
        self.c = rat
        self.d = oldr
    def reset(self):
        ## random inititalize
        self.state_note[:,:,:] = np.eye(vecLen)[np.random.choice(vecLen, segLen)]
        self.state_delta[:,:,:]= np.eye(maxdelta)[np.random.choice(maxdelta, segLen)]
        self.firstNote = None
    def countFinger(self, x, y, deltas, notes, lim):
        if x>0: return 0
        cnt=1 ## self
        for i, v in enumerate(reversed(deltas)): ## others
            cnt += 1 if self.sameTrack(y, notes[segLen-1-i]) else 0
            if v!=0: break
        return 0 if cnt<=lim else -(cnt-lim)*2
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
            return -6
        if diffLastNote==4: ## western
            return 3
        elif diffLastNote==8:
            return 2
        elif diffLastNote==7: ## chinese
            return -4
        elif diffLastNote==5:
            return -3
        elif diffLastNote<=2 and delta==0: ## annoying sound
            return -5
        elif diffLastNote==12 and delta==0: ## full 8
            return 1
        return 0
    def sameTrack(self, a, b):
        return (a<pianoKeys and b<pianoKeys) or (a>=pianoKeys and b>=pianoKeys)
    def checkAccompanyDist(self, delta, notes, deltas):
        accumTick=delta
        i = None
        for ti, v in enumerate(reversed(deltas)):
            i = segLen-1-ti
            if notes[i]>=pianoKeys: break ## find accompany
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
        with tf.device('/cpu:0'):
            p_n, p_d = self.rewardRNN.predict([self.state_note, self.state_delta], verbose=0)
        pitchStyleReward = math.log(p_n[0][action_note])
        tickStyleReward = math.log(p_d[0][action_delta])
        reward_note=0
        reward_delta=0
        if np.sum(self.state_note)==segLen and np.sum(self.state_delta)==segLen:
            state_idx_note = [ np.where(r==1)[0][0] for r in self.state_note[0] ]
            state_idx_delta = [ np.where(r==1)[0][0] for r in self.state_delta[0] ]
            if self.countSameNote(action_note, state_idx_note)>4:
                done = True ## bad end
            if action_note<pianoKeys: ## is main melody
                dist, idx = self.checkAccompanyDist(action_delta, state_idx_note, state_idx_delta)
                if dist>64: ## 一小節
                    reward_delta -= 8 ## too far
                if not idx is None: ## idx points to a nearest accompany note
                    ## find root note
                    rootN = self.findRootNote(idx, state_idx_note, state_idx_delta)
                    dist = abs(rootN-action_note)%12
                    if dist==0: ## check if valid chord
                        reward_note += 3
                    elif dist==8:
                        reward_note += 2
                    elif dist==4:
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
            if diff>0: ## check update
                if verbose:
                    sys.stderr.write('lrs changed: '+str(diff)+'\n')
                reward_note += 2*diff
            if lrsNote_new>8 or lrsNote_new==0:
                done = True ## bad end, very bad...
            ## not complete yet...
            if action_note<pianoKeys: ## main
                reward_delta += self.countFinger(action_delta, action_note, state_idx_delta, state_idx_note, 5)
            else: ## accompany
                reward_delta += self.countFinger(action_delta, action_note, state_idx_delta, state_idx_note, 4)

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
        reward_note = -20. if done else reward_note*self.c+self.d*pitchStyleReward
        reward_delta= reward_delta*self.c+self.d*tickStyleReward
        return reward_note, reward_delta, done


if __name__ == "__main__":
    agent = DQNAgent(policy='softmax', verbose=True)
    agent.load(str(sys.argv[1]))
    rewardSys = rewardSystem(0.05,0.1) ## more sensitive
    done = False
    batch_size = 64
    batch_n = 16

    with open('./log.csv', 'a+', 0) as logFP: ## no-buffer logging
        logFP.write('pitch, tick\n')
        rewardSys.reset() ## initialize states
        for e in xrange(EPISODES):
            snote, sdelta = rewardSys.get_state() ## give initial state
            for time in xrange(64):
                action_note, action_delta = agent.act([snote, sdelta]) ## action on state
                reward_note, reward_delta, done = rewardSys.reward(action_note, action_delta, verbose=False) ## reward on state
                reward_note = np.clip(reward_note, -1, 1)
                reward_delta = np.clip(reward_delta, -1, 1)
                if time % 4 == 0:
                    logFP.write('%.2f, %.2f\n' % (reward_note, reward_delta))
                nnote, ndelta = rewardSys.get_state() ## get next state
                agent.remember(snote, sdelta, action_note, action_delta, reward_note, reward_delta, nnote, ndelta, done)
                snote, sdelta = nnote, ndelta ## update current state
                if done: ## termination
                    agent.update_target_model() ## update
                    sys.stderr.write('Target network has been updated. Reset stage.\n')
                    rewardSys.reset() ## new initial state
                    break
            if len(agent.memory) > batch_size:
                sys.stderr.write('episode: %d Learning from past...' % e)
                for t in xrange(batch_n): ## replay for batch_n times
                    agent.replay(batch_size)
                sys.stderr.write(', eps: %.2f\n' % agent.decay())
                if e % 10 == 0:
                    agent.save("./save/melody-ddqn-{}.h5".format(e))
                    agent.update_target_model() ## force update
