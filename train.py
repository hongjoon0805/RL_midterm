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
from network import DuelModel
tf.keras.backend.set_floatx('float64')
    
# 지하철 안에서 distributional + per 까지 모두 짜보자. 디버깅을 빠르게 진행해야함
# rainbow 구현 모두 끝나면 reward 체계를 바꿔보자. bar에 닿는 횟수가 매우 적으니, bar에 닿을 때 마다 reward를 100정도 얻을 수 있게 해보자.
class DQN:
    def __init__(self, mode, random = False):
        
        self.mode_idx = 0 if mode == 'l' else 1
            
        self.state_size = 10
        self.action_size = 3
        self.random = random
        
        self.memory_size  = int(2e4)
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
        self.support = tf.cast(tf.linspace(self.v_min, self.v_max, self.atom_size), dtype=tf.float64)
        
        # N-step Learning
        self.n_step = 3
        
        # PER
        # memory for 1-step Learning
        self.memory = PrioritizedReplayBuffer(self.state_size, self.memory_size, self.batch_size, alpha=self.alpha)
        
        # memory for N-step Learning
        self.memory_n = ReplayBuffer(self.state_size, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma)
        
        self.model = DuelModel(self.state_size, self.action_size, self.atom_size)
        self.target_model = DuelModel(self.state_size, self.action_size, self.atom_size)
        
        self.update_target_model()
        
        self.transition = list()
        
        self.frame_cnt = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000125)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    
    def update_model(self):
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = samples["weights"].reshape(-1, 1)
        indices = samples["indices"]
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        
        gamma = self.gamma ** self.n_step
        samples = self.memory_n.sample_batch_from_idxs(indices)
        
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"].reshape(-1, 1)
        done = samples["done"].reshape(-1, 1)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        
        with tf.GradientTape() as tape:
            
            elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            loss = tf.math.reduce_mean(elementwise_loss * weights)
        
        model_weights = self.model.trainable_variables
        # PER: importance sampling before average
        grad = tape.gradient(loss, model_weights)
        self.optimizer.apply_gradients(zip(grad, model_weights))
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
    
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float):
        """Return categorical dqn loss."""
        
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"].reshape(-1, 1)
        done = samples["done"].reshape(-1, 1)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        # Double DQN
        dist = self.model(next_state)
        q = tf.math.reduce_sum(dist, axis=2)
        next_action = tf.math.argmax(q, axis=1)
        next_dist = self.target_model(next_state)
        
        index = tf.stack([tf.range(self.batch_size, dtype=tf.int64), next_action], axis=0)
        index = tf.transpose(index)
        next_dist = tf.cast(tf.gather_nd(next_dist, index), dtype=tf.float64)

        t_z = reward + (1 - done) * gamma * self.support
        t_z = tf.clip_by_value(t_z, self.v_min, self.v_max)
        b = tf.cast((t_z - self.v_min) / delta_z, dtype=tf.float64)
        l = tf.cast(tf.math.floor(b), dtype=tf.int64)
        u = tf.cast(tf.math.ceil(b), dtype=tf.int64)

        offset = tf.linspace(0.0, float((self.batch_size - 1) * (self.atom_size )), self.batch_size)
        print(offset)
        offset = tf.cast(offset,dtype=tf.int64)
        offset = tf.expand_dims(offset, 1)
        offset = tf.broadcast_to(offset, [self.batch_size, self.atom_size])

        proj_dist = tf.reshape(tf.zeros(tf.shape(next_dist), dtype=tf.float64), [-1])
        
        proj_dist = tf.tensor_scatter_nd_add(proj_dist, 
                                             tf.reshape(l + offset, [-1,1]),
                                             tf.reshape(next_dist * (tf.cast(u, tf.float64) - b), [-1]))

        proj_dist = tf.tensor_scatter_nd_add(proj_dist, 
                                             tf.reshape(u + offset, [-1,1]),
                                             tf.reshape(next_dist * (b - tf.cast(l, tf.float64)), [-1]))
        
        proj_dist = tf.reshape(proj_dist, tf.shape(next_dist))

        dist = self.model(state)
        # indexing은 tf.gather_nd 사용하자 <-- 이거 조심. 디버깅 할 때 문제 있으면 이부분 부터 보자.
        action = tf.convert_to_tensor(action, dtype = tf.int64)
        index = tf.stack([tf.range(self.batch_size, dtype=tf.int64), action], axis=0)
        index = tf.transpose(index)
        log_p = tf.math.log(tf.gather_nd(dist, index))
        elementwise_loss = -tf.math.reduce_sum(proj_dist * log_p, axis = 1)

        return elementwise_loss
    
    def _cal_reward(self, state, reward):
        bar_x, bar_y = state[:2]
        ball_x, ball_y = state[2:4]
        ball_radius = 0.025
        bar_radius = 0.05
        bar_coord = [bar_x-0.05, bar_x-0.025, bar_x, bar_x+0.025, bar_x+0.05]
        if abs(bar_y - ball_y) == ball_radius:
            if ball_x in bar_coord:
                reward = 10

        elif bar_y==ball_y:
            reward = -1
        
        return reward
    
    def pre_process(self, next_state: np.ndarray, reward, done, frame_cnt, num_frames) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        reward = self._cal_reward(next_state, reward)
        
        self.transition += [reward, next_state, done]

        # N-step transition
        one_step_transition = self.memory_n.store(*self.transition)

        # add a single step transition
        if one_step_transition:
            self.memory.store(*one_step_transition)
        
        # PER: increase beta
        fraction = min(frame_cnt / num_frames, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)
    
        return next_state, reward, done
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        
        if self.random:
            return np.random.choice(range(self.action_size), 1, replace=False)[0]
        
        dist = self.model(state)
        
        actions = tf.math.reduce_sum(dist * self.support, axis=2)
        selected_action = np.argmax(actions)
        
        self.transition = [state, selected_action]
        
        return selected_action
    
#     def train(self, episodes: int):
#         """Train the agent."""
        
#         state = self.env.reset()
#         frame_cnt = 0
#         num_frames = 5e6
#         for ep_i in range(episodes):
#             done_n = [False for _ in range(self.env.n_agents)]
#             self.env.seed(ep_i)
#             state = np.array(self.env.reset())
#             rewards_cnt = np.array([0,0])
            
#             if frame_cnt > num_frames:
#                 break

#             while not all(done_n): 
#                 action = self.select_action(state)
#                 next_state, reward, done = self.step(action)
#                 state = next_state
                
#                 # PER: increase beta
#                 fraction = min(frame_cnt / num_frames, 1.0)
#                 self.beta = self.beta + fraction * (1.0 - self.beta)

#                 rewards_cnt[0] += reward_n[0]
#                 rewards_cnt[1] += reward_n[1]
                
#                 # if training is ready
#                 if len(self.memory) >= self.batch_size:
#                     self.update_model()
                    
#                 frame_cnt += 1
                

#             self.update_target_model()

#             last_100_episode[0].append(rewards_cnt[0])
#             last_100_episode[1].append(rewards_cnt[1])


#             print('Episode:%d || Left: %d || Right: %d || Left Avg: %.2f || Right Avg: %.2f'%(ep_i, 
#                                                                           rewards_cnt[0], 
#                                                                           rewards_cnt[1],                                  
#                                                                           np.mean(last_100_episode[0]), 
#                                                                           np.mean(last_100_episode[1]),))
            

                
#         self.env.close()
    

env = gym.make('PongDuel-v0')
dqn = [DQN('l', False), DQN('r', True)]

f = open('log.txt', 'w')

turn = 0

last_100_episode = [deque(maxlen=100), deque(maxlen=100)]


state = env.reset()
frame_cnt = 0
num_frames = 5e6
episodes = int(1e5)
for ep_i in range(episodes):
    done_n = [False for _ in range(env.n_agents)]
    env.seed(ep_i)
    state = np.array(env.reset())
    rewards_cnt = np.array([0,0], dtype=np.float64)

    if frame_cnt > num_frames:
        break

    while not all(done_n): 
        action = [dqn[0].select_action(state[0].reshape(1,10)), dqn[1].select_action(state[1].reshape(1,10))]
        next_state_n, reward_n, done_n, info = env.step(action)
        next_state_n = np.array(next_state_n, dtype=np.float64)
        if all(done_n):
            ball_x, ball_y = next_state_n[0][2:4]
            if ball_y < 0.5:
                reward_n[0] += 1
            else:
                reward_n[1] += 1
        rewards_cnt += np.array(reward_n)
        next_state, reward, done = next_state_n[turn], reward_n[turn], done_n[turn]
        next_state, reward, done = dqn[turn].pre_process(next_state, reward, done, frame_cnt, num_frames)
        state = next_state_n

        # if training is ready
        if len(dqn[turn].memory) >= dqn[turn].batch_size:
            dqn[turn].update_model()

        frame_cnt += 1


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
f.close()