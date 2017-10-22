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
    def __init__(self, lr=1e-7, gamma=0.95, model=None): ## learning_rate, gamma, model_to_be_tuned
        self.learning_rate = lr
        self.model = self.build(model) ## warn
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

    def _build_model(self, model):
        # Neural Net for PG learning Model
        ## re-complie the model to ensure that the loss is categorical-crossentropy
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

    def save(self, name):
        self.model.save(name)

class rewardSystem:
    def __init__(self, model_dir=None):
        self.rewardRNN = None
        if not model_dir is None:
            self.rewardRNN = [ (load_model(str(model_dir)+'/'+r), self.fn2float(r)) for r in os.listdir(str(model_dir)) ]
        self.state_note = np.zeros((1, segLen, vecLen), dtype=np.bool)
        self.state_delta= np.zeros((1, segLen, maxdelta), dtype=np.bool)
        self.firstNote = None
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
    def reward(self, action_note, action_delta, verbose=False):
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

        ## update state:
        self.state_note = np.roll(self.state_note, -1, axis=1)
        self.state_note[0,-1,:] = 0
        self.state_note[0,-1,action_note] = 1

        self.state_delta = np.roll(self.state_delta, -1, axis=1)
        self.state_delta[0,-1,:] = 0
        self.state_delta[0,-1,action_delta] = 1
        return pitchStyleReward, tickStyleReward


if __name__ == "__main__":
    agent = PGAgent(lr=1e-7, gamma=0.99, model=load_model(str(sys.argv[1])))
    seedPos = str(sys.argv[2])
    rewardSys = rewardSystem(model_dir = str(sys.argv[3])) ## more sensitive
    batch_size = 64

    with open('./pg.csv', 'a+', 0) as logFP: ## no-buffer logging
        logFP.write('pitch, tick\n')
        rewardSys.reset(seed=seedPos) ## initialize states
        score_note = 0.
        score_delta = 0.
        for e in xrange(EPISODES):
            rewardSys.reset(seed=seedPos) ## new initial state
            snote, sdelta = rewardSys.get_state() ## give initial state
            for time in xrange(batch_size):
                action_note, action_delta, p_n, p_d = agent.act([snote, sdelta]) ## action on state
                reward_note, reward_delta = rewardSys.reward(action_note, action_delta, verbose=False) ## reward on state
                score_note += float(reward_note)
                score_delta += float(reward_delta)
                agent.remember(action_note, action_delta, snote, sdelta, float(reward_note), float(reward_delta), p_n, p_d)
                snote, sdelta = rewardSys.get_state() ## get next state
            sys.stderr.write('episode: %d Learning from past... bs: %d\n' % (e, len(agent.notes)))
            logFP.write("%.2f, %.2f\n" % (score_note, score_delta))
            agent.train()
            if e % 10 == 0:
                agent.save("./pg/melody-ddqn-{}.h5".format(e))
