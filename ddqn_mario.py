# -*- coding: utf-8 -*-
'''
DQN architecture
Folk from: https://github.com/keon/deep-q-learning
'''
from __future__ import print_function
import os
import sys
import gym
import gym_pull
#gym_pull.pull('github.com/ppaquette/gym-super-mario')
import random
import numpy as np
from collections import deque
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, concatenate, Dropout, BatchNormalization, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import load_model

EPISODES = 10000

def preprocess(x):
    x = x.astype(np.float32) ## ?-type -> float32
    x = np.expand_dims(x, 0)
    x[...,0] -= 123.68 #R
    x[...,1] -= 116.779 #G
    x[...,2] -= 103.939 #B
    return x

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
    def __init__(self, policy='softmax', verbose=False, action_num=None, lr=0.001):
        self.memory = deque(maxlen=3072)
        self.state_shape=(224, 256, 3)
        self.gamma = 0.8    # discount rate
        self.epsilon = 0.99  # exploration rate
        self.epsilon_min = 0.001 ## large eps
        self.epsilon_decay = 0.99
        self.learning_rate = lr
        self.action_num = action_num
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
        model = Sequential()
        model.add(Conv2D(32, 7, strides=(2,2), padding='same', data_format='channels_last', activation='relu', kernel_initializer='RandomUniform', input_shape=self.state_shape))
        model.add(Conv2D(64, 7, strides=(2,2), padding='same', data_format='channels_last', activation='relu', kernel_initializer='RandomUniform'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='RandomUniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='RandomUniform'))
        model.add(Dense(self.action_num, activation='linear', kernel_initializer='RandomUniform'))
        model.compile(loss=self._huber_loss,
                      optimizer=RMSprop(lr=self.learning_rate, clipnorm=1.))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): ## Using CPU
        if self.policy=='softmax': ## softmax policy
            act = self.model.predict(state)
            return softmaxSample(act[0])
        else: ## epsilon-greedy
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_num)
            act = self.model.predict(state)
            return np.argmax(act[0])

    def replay(self, batch_size): ## Using GPU
        minibatch = random.sample(self.memory, batch_size)
        batch = np.zeros((batch_size, 224, 256, 3), dtype=np.bool)
        #state(0), action(1), reward(2), next_state(3), done(4) = entries
        for i, entries in enumerate(minibatch):
            batch[i,:,:,:] = entries[0][0] ## state_note
        state = batch ## a batch of state
        target = self.model.predict(batch) ## get a batch of target
        for i, entries in enumerate(minibatch):
            batch[i,:,:,:] = entries[3][0] ## next_state
        a = self.model.predict(batch) ## get a batch of q-value of new model
        t = self.target_model.predict(batch) ## old model
        for i, entries in enumerate(minibatch):
            target[i][entries[1]] = entries[2] + (self.gamma*t[i][np.argmax(a[i])] if not entries[4] else 0) 
        self.model.train_on_batch(state, target) ## a minibatch

    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return self.epsilon

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    BATCH_SIZE = 256
    BATCH_N = 16
    skip = 2 ## frame skip
    #env = gym.make("ppaquette/SuperMarioBros-1-1-v0")
    #gym from: https://github.com/ppaquette/gym-super-mario
    env = gym.make("SuperMarioBros-1-1-v0")
    action_size = 6
    agent = DQNAgent(lr=0.01, action_num=action_size)
    if os.path.isfile('./Mario.h5'):
        agent.load('./Mario.h5')
    state = preprocess(env.reset()) ## get initial state
    env.render() ## open screen
    with open('./score.csv', 'a+', 0) as logFP: ## no-buffer logging
        for e in xrange(EPISODES):
            done = False
            score = 0
            step = 0
            while True:
                act = agent.act(state) ## action on state
                act_array = [0]*6
                act_array[act] = 1
                nstate, reward, done, info = env.step(act_array)
                score += reward
                reward = np.clip(reward, -1, 1) ## for better convergence
                if step%skip==0 or done:
                    agent.remember(state, act, reward, nstate, done)
                state = preprocess(nstate) ## next state
                step += 1
                if done: ## termination
                    logFP.write('%d\n' % score)
                    agent.update_target_model() ## update
                    sys.stderr.write('Target network has been updated. Reset stage.\n')
                    env.change_level(0)
                    break
            if len(agent.memory) > BATCH_SIZE:
                sys.stderr.write('episode: %d Learning from past... bs: %d\n' % (e, len(agent.memory)))
                for t in xrange(BATCH_N):
                    agent.replay(batch_size=BATCH_SIZE)
                sys.stderr.write(', eps: %.2f\n' % agent.decay())
                if e % 10 == 0:
                    agent.save("./ddqn-{}.h5".format(e))
    agent.save('./Mario.h5')
