# -*- coding: utf-8 -*-
import sys
import random
import numpy as np
from collections import deque
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
from attention_block import SoftAttentionBlock

EPISODES = 5000
segLen=48
track_num=2
pianoKeys=60
vecLen=pianoKeys*track_num
maxdelta=33
hidden_delta=128
hidden_note=256
drop_rate=0.2

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

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
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state_note, state_delta, action_note, action_delta, reward_note, reward_delta,  next_state_note, next_state_delta, done):
        self.memory.append((state_note, state_delta, action_note, action_delta, reward_note, reward_delta,  next_state_note, next_state_delta, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(vecLen), random.randrange(maxdelta)
        act_note, act_delta = self.model.predict(state)
        return np.argmax(act_note[0]), np.argmax(act_delta[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        batch_note = np.zeros((batch_size, segLen, vecLen), dtype=np.bool)
        batch_delta= np.zeros((batch_size, segLen, maxdelta), dtype=np.bool)
        batch_nnote= np.zeros((batch_size, vecLen), dtype=np.bool)
        batch_ndelta=np.zeros((batch_size, maxdelta), dtype=np.bool)
        for i, entries in enumerate(minibatch):
            state_note, state_delta, action_note, action_delta, reward_note, reward_delta,  next_state_note, next_state_delta, done = entries
            target_note, target_delta = self.model.predict([state_note, state_delta])
            if done:
                target_note[0][action_note] = reward_note
                target_delta[0][action_delta] = reward_delta
            else:
                a_note, a_delta = self.model.predict([next_state_note, next_state_delta])
                t_note, t_delta = self.target_model.predict([next_state_note, next_state_delta])
                target_note[0][action_note] = reward_note + self.gamma * t_note[0][np.argmax(a_note[0])]
                target_delta[0][action_delta] = reward_delta + self.gamma * t_delta[0][np.argmax(a_delta[0])]

            batch_note[i,:,:] = state_note[0]
            batch_delta[i,:,:]= state_delta[0]
            batch_nnote[i,:]= target_note[0]
            batch_ndelta[i,:]=target_delta[0]
        self.model.fit([batch_note, batch_delta], [batch_nnote, batch_ndelta], epochs=1, verbose=0) ## a minibatch
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
    def __init__(self, rat):
        self.rewardRNN = load_model("./NoteRNN.h5")
        self.state_note = np.zeros((1, segLen, vecLen), dtype=np.bool)
        self.state_delta= np.zeros((1, segLen, maxdelta), dtype=np.bool)
        self.firstNote = None
        self.c = rat
    def reset(self):
        self.state_note = np.zeros((1, segLen, vecLen), dtype=np.bool)
        self.state_delta= np.zeros((1, segLen, maxdelta), dtype=np.bool)
        self.firstNote = None
    def countFinger(self, x, deltas):
        if x>0: return 0
        cnt=2
        for v in reversed(deltas):
            if v==0:
                cnt+=1
            else: break
        return 0 if cnt<=4 else -cnt*10
    def countSameNote(self, x, l):
        cnt = 1
        for v in reversed(l):
            if v==x: cnt+=1
            else: break
        return -10*(cnt-4) if cnt>4 else 0
    def get_state(self):
        return self.state_note, self.state_delta
    def scale(self, diffLastNote):
        if diffLastNote==4: ## western
            return 4
        elif diffLastNote==8:
            return 2
        #elif diffLastNote==12:
            #return 1
        elif diffLastNote==7: ## chinese
            return -3
        elif diffLastNote==8:
            return -4
        return 0
    def reward(self, action_note, action_delta):
        done = False
        p_n, p_d = self.rewardRNN.predict([self.state_note, self.state_delta], verbose=0)
        pitchStyleReward = math.log(p_n[0][action_note])
        tickStyleReward = math.log(p_d[0][action_delta])
        reward_note=0
        reward_delta=0
        if np.sum(self.state_note)==segLen and np.sum(self.state_delta)==segLen:
            state_idx_note = [ np.where(r==1)[0][0] for r in self.state_note[0] ]
            state_idx_delta = [ np.where(r==1)[0][0] for r in self.state_delta[0] ]
            if not self.firstNote is None and abs(self.firstNote-action_note)%12==0:
                done = True
                reward_note+=10
            reward_note += self.countSameNote(action_note, state_idx_note)
            ## determine
            diffLastNote = abs(action_note - state_idx_note[-1])
            reward_note += self.scale(diffLastNote)
            lrsi = state_idx_note
            lrsNote_old = lrs(lrsi)
            lrsi.append(action_note)
            lrsNote_new = lrs(lrsi)
            if lrsNote_new>=16:
                reward_note-=50
            else: ## <16
                reward_note += 10*(lrsNote_new- lrsNote_old)
            reward_delta += self.countFinger(action_delta, state_idx_delta)*10

        self.state_note = np.roll(self.state_note, -1, axis=1)
        self.state_note[0,-1,:] = 0
        self.state_note[0,-1,action_note] = 1

        self.state_delta = np.roll(self.state_delta, -1, axis=1)
        self.state_delta[0,-1,:] = 0
        self.state_delta[0,-1,action_delta] = 1
        if self.firstNote is None:
            self.firstNote = action_note
        return reward_note*self.c+pitchStyleReward, reward_delta*self.c+tickStyleReward, done


if __name__ == "__main__":
    agent = DQNAgent()
    agent.load(str(sys.argv[1]))
    rewardSys = rewardSystem(0.9)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        rewardSys.reset()
        snote, sdelta = rewardSys.get_state()
        tns = 0 ## total pitch score
        tds = 0 ## total tick score
        for time in range(500):
            action_note, action_delta = agent.act([snote, sdelta])
            reward_note, reward_delta, done = rewardSys.reward(action_note, action_delta)
            tns += reward_note
            tds += reward_delta
            nnote, ndelta = rewardSys.get_state()
            agent.remember(snote, sdelta, action_note, action_delta, reward_note, reward_delta, nnote, ndelta, done)
            snote, sdelta = nnote, ndelta
            if done:
                agent.update_target_model()
                sys.stderr.write("episode: {}/{}, time: {}, e: {:.2}, avg_pitch_score: {:.2}, avg_tick_score: {:.2}\n"
                      .format(e, EPISODES, time, agent.epsilon, tns/time, tds/time))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("./save/melody-ddqn-{}-{:.2}-{:.2}.h5".format(e, tns/time, tds/time))
