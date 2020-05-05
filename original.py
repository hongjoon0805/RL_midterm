   
import gym
from ma_gym.wrappers import Monitor
import math
import os
import random
import sys

import numpy as np
import tensorflow.compat.v1 as tf
from collections import deque
import operator
from memory import PrioritizedReplayBuffer, ReplayBuffer
from typing import Deque, Dict, List, Tuple, Callable
from network import DuelModel, OldDuelModel
tf.keras.backend.set_floatx('float64')
    
# 지하철 안에서 distributional + per 까지 모두 짜보자. 디버깅을 빠르게 진행해야함
# rainbow 구현 모두 끝나면 reward 체계를 바꿔보자. bar에 닿는 횟수가 매우 적으니, bar에 닿을 때 마다 reward를 100정도 얻을 수 있게 해보자.
class DQN:
    def __init__(self, random = False):

        self.state_size = 10
        self.action_size = 3
        self.random = random
        
        self.memory_size  = int(1e5)
        self.batch_size = 128
        self.gamma = 0.99
        
        # PER parameters
        self.alpha = 0.2
        self.beta = 0.6
        self.prior_eps = 1e-6
        
        # Categorical DQN parameters
        self.v_min = 0.0
        self.v_max = 200.0
        self.atom_size = 51
        
        # N-step Learning
        self.n_step = 3
        
        # PER
        # memory for 1-step Learning
        self.memory = PrioritizedReplayBuffer(self.state_size, self.memory_size, self.batch_size, alpha=self.alpha)
        
        # memory for N-step Learning
        self.memory_n = ReplayBuffer(self.state_size, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma)
        
        self.model = OldDuelModel(self.state_size, self.action_size)
        self.target_model = OldDuelModel(self.state_size, self.action_size)
        
        self.update_target_model()
        self.buffer_dict = {'state':deque(maxlen=self.memory_size), 
                       'action':deque(maxlen=self.memory_size), 
                       'reward':deque(maxlen=self.memory_size),
                       'next_state':deque(maxlen=self.memory_size),
                       'done':deque(maxlen=self.memory_size)}
        
        self.frame_cnt = 0
        self.optimizer = tf.keras.optimizers.Adam()

    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환
        
        if self.random:
            action = np.random.choice(range(self.action_size), 1, replace=False)
            return action
        
        return np.argmax(self.model(state))
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def train_minibatch(self, X, Y):
        # mini batch를 받아 policy를 update
        
        with tf.GradientTape() as tape:
            loss = tf.math.reduce_mean(tf.square(Y - self.model(X)))
        model_vars = self.model.trainable_variables
        grad = tape.gradient(loss, model_vars)
        self.optimizer.apply_gradients(zip(grad, model_vars))
        
        self.model.reset_noise()
        self.target_model.reset_noise()
        
    def update_epsilon(self):
        # Exploration 시 사용할 epsilon 값을 업데이트
        
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        
        return
    
    def update_buffer(self, state, action, reward, next_state, done):
        self.buffer_dict['state'].append(state)
        self.buffer_dict['action'].append(action)
        self.buffer_dict['reward'].append(reward)
        self.buffer_dict['next_state'].append(next_state)
        self.buffer_dict['done'].append(int(done))
    
    
    def make_minibatch(self,buffer_dict):
        bsize = self.batch_size
        batch_idx = np.random.choice(list(range(len(buffer_dict['state'])-(self.n_step-1))), bsize, replace=False)
            
        states = np.array(buffer_dict['state'])[batch_idx]
        actions = np.array(buffer_dict['action'])[batch_idx]
        rewards_arr = np.zeros((bsize,self.n_step))
        gamma_arr = np.ones((self.n_step,))
        for i in range(self.n_step):
            rewards_arr[:,i] = np.array(buffer_dict['reward'])[batch_idx+i]
            if i > 0:
                gamma_arr[i] = self.gamma * gamma_arr[i-1]

        rewards = (rewards_arr * gamma_arr).sum(axis=1)
            
        next_states = np.array(buffer_dict['next_state'])[batch_idx + (self.n_step-1)]
        dones = np.array(buffer_dict['done'])[batch_idx]
        
        # Double
        action_target = self.model(states).numpy()
        action_max = np.argmax(action_target, axis=1)
        
        q_value = self.target_model(next_states).numpy()[np.arange(bsize), action_max]
        q_target_out = rewards + (self.gamma ** self.n_step) * q_value
        target = q_target_out * (1-dones) + rewards * (dones)
        action_target[np.arange(bsize), actions] = target
        
        X, Y = states, action_target
        
        return X, Y

env = gym.make('PongDuel-v0')

f = open('log_original.txt','w')

dqn1 = DQN(False)
dqn2 = DQN(True)

dqn = [dqn1, dqn2]
turn = 0

last_100_episode = [deque(maxlen=100), deque(maxlen=100)]

def _cal_reward(state, reward):
    bar_x, bar_y = state[:2]
    ball_x, ball_y = state[2:4]
    ball_radius = 0.025
    bar_radius = 0.05
    bar_coord = [bar_x-0.05, bar_x-0.025, bar_x, bar_x+0.025, bar_x+0.05]
    if abs(bar_y - ball_y) == ball_radius:
        if ball_x in bar_coord:
            reward = 0.1

    elif bar_y==ball_y:
        reward = -1

    return reward

for ep_i in range(10000):
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0
    env.seed(ep_i)
    state = np.array(env.reset())
    rewards_cnt = np.array([0,0])
    
    while not all(done_n): 
        action = [dqn1.predict(state[0].reshape(1,10)), dqn2.predict(state[1].reshape(1,10))]
        next_state, reward_n, done_n, info = env.step(action)
        next_state = np.array(next_state)
        
        if all(done_n):
            ball_x, ball_y = next_state[0][2:4]
            if ball_y > 0.5:
                reward_n[0] += 1
            else:
                reward_n[1] += 1
        
        rewards_cnt[0] += reward_n[0]
        rewards_cnt[1] += reward_n[1]
        reward_n[0] = _cal_reward(next_state[0], reward_n[0])
        
        dqn[turn].update_buffer(state[turn], action[turn], reward_n[turn], next_state[turn], done_n[turn])
        if dqn[turn].frame_cnt > 150:
            X, Y = dqn[turn].make_minibatch(dqn[turn].buffer_dict)
            dqn[turn].train_minibatch(X,Y)
#             dqn[turn].update_epsilon()
        dqn[turn].frame_cnt += 1
        state = next_state
        
    dqn[turn].update_target_model()
    
    
    last_100_episode[0].append(rewards_cnt[0])
    last_100_episode[1].append(rewards_cnt[1])
    
    
    print('Episode:%d || Left: %d || Right: %d || Left Avg: %.2f || Right Avg: %.2f'%(ep_i, 
                                                                  rewards_cnt[0], 
                                                                  rewards_cnt[1],                                  
                                                                  np.mean(last_100_episode[0]), 
                                                                  np.mean(last_100_episode[1]),))
    
    f.write('Episode:%d || Left: %d || Right: %d || Left Avg: %.2f || Right Avg: %.2f\n'%(ep_i, 
                                                                  rewards_cnt[0], 
                                                                  rewards_cnt[1],                                  
                                                                  np.mean(last_100_episode[0]), 
                                                                  np.mean(last_100_episode[1]),))
    
    if (ep_i+1) % 10000 == 0:
        turn ^= 1
