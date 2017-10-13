# -*- coding: utf-8 -*-
'''
DQN architecture
Folk from: https://github.com/keon/deep-q-learning
'''
from __future__ import print_function
import os
import sys
import gym
import random
import numpy as np
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, concatenate, Dropout, BatchNormalization, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import load_model

EPISODES = 8000

class PGAgent:
    def __init__(self, lr=1e-6, gamma=0.95, state_shape=(210, 160, 3), action_num=None): ## low lr to tune all weights
        self.learning_rate = lr
        self.state_shape = state_shape
        self.action_num = action_num
        self.model = self._build_model()
        self.state=[]
        self.prob=[]
        self.grad  =[]
        self.reward=[]
        self.gamma = gamma
    def reset(self):
        self.state, self.prob, self.grad, self.reward = [],[],[],[]
    def _build_model(self):
        # Neural Net for PG learning Model
        model = Sequential()
        model.add(Conv2D(32, 3, padding='same', data_format='channels_last', activation='relu', input_shape=self.state_shape))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, 3, padding='same', data_format='channels_last', activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(self.action_num, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=self.learning_rate, clipnorm=1.))
        return model

    def remember(self, act, state, reward, prob):
        ## appends all:
        self.state.append(state)
        self.reward.append(reward)
        self.prob.append(prob)
        new_p = np.zeros(self.action_num, dtype=np.float32)
        new_p[act] = 1
        self.grad.append(new_p - prob)

    def act(self, state): ## Using CPU
        state = np.expand_dims(state, 0)
        act = self.model.predict(state)
        act = act[0] / np.sum(act[0]) ## norm -> PMF
        return np.random.choice(self.action_num, 1, p=act)[0], act
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

    def train(self, batch_size=128): ## Using GPU
        _grad = np.vstack(self.grad)
        _reward = np.vstack(self.reward)
        _grad *= self._discount_and_norm_rewards(_reward)
        _state = np.array(self.state)
        target = np.vstack(self.prob) + self.learning_rate * _grad
        self.model.fit(_state, target, batch_size=batch_size, verbose=0)
        self.reset() ## forget it

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    BATCH_SIZE = 32
    env = gym.make("MsPacman-v0")
    action_size = env.action_space.n
    agent = PGAgent(action_num=action_size)
    if os.path.isfile('./Pacman.h5'):
        agent.load('./Pacman.h5')

    with open('./pg.csv', 'a+', 0) as logFP: ## no-buffer logging
        logFP.write('score\n')
        for e in xrange(EPISODES):
            done = False
            state = env.reset() ## new game
            score = 0
            while not done:
                env.render()
                act, p = agent.act(state) ## action on state
                nstate, reward, done, info = env.step(act)
                score += reward
                agent.remember(act, state, reward, p)
                state = nstate
                if done: ## termination
                    sys.stderr.write('episode: %d Learning from past... bs: %d\n' % (e, len(agent.state)))
                    logFP.write('%d\n' % score)
                    agent.train(batch_size=BATCH_SIZE)
                    break
            if e % 10 == 0:
                agent.save("./pg-{}.h5".format(e))
    agent.save('./Pacman.h5')
