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
    def __init__(self, random = False):

        self.state_size = 10
        self.action_size = 3
        self.random = random
        
        self.memory_size  = 1e4,
        self.batch_size = 128,
        self.gamma = 0.99,
        
        # PER parameters
        self.alpha = 0.2,
        self.beta = 0.6,
        self.prior_eps = 1e-6,
        
        # Categorical DQN parameters
        self.v_min = 0.0,
        self.v_max = 200.0,
        self.atom_size = 51,
        
        # N-step Learning
        self.n_step = 3,
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(self.state_size, self.memory_size, self.batch_size, alpha=self.alpha)
        
        # memory for N-step Learning
        self.memory_n = ReplayBuffer(self.state_size, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma)
        
        self.model = DuelModel(self.state_size, self.action_size, self.atom_size)
        self.target_model = DuelModel(self.state_size, self.action_size, self.atom_size)
        
        self.update_target_model()
        
        self.transition = list()
        
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
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        self.transition += [reward, next_state, done]

        # N-step transition
        one_step_transition = self.memory_n.store(*self.transition)

        # add a single step transition
        self.memory.store(*one_step_transition)
    
        return next_state, reward, done
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        dist = self.dqn(state)
        
        action = tf.math.reduce_sum(dist * self.support, dim=2).numpy().argmax()
        
        selected_action = selected_action.detach().cpu().numpy()
        
        self.transition = [state, selected_action]
        
        return selected_action
    
    def update_model(self) -> torch.Tensor:
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

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)
        
        model_weights = self.dqn.trainable_variables
        grad = tape.gradient(loss, model_weights)
        self.optimizer.apply_gradients(zip(grad, model_weights))
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
    
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"].reshape(-1, 1)
        done = samples["done"].reshape(-1, 1)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        # Double DQN
        next_action = self.dqn(next_state).argmax(1)
        next_dist = self.dqn_target.dist(next_state)
        next_dist = next_dist[range(self.batch_size), next_action]

        t_z = reward + (1 - done) * gamma * self.support
        t_z = tf.clip_by_value(self.v_min, self.v_max)
        b = (t_z - self.v_min) / delta_z
        l = tf.cast(tf.math.floor(b), dtype=tf.float64)
        u = tf.cast(tf.math.ceil(b), dtype=tf.float64)

        offset = tf.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size)
        offset = tf.cast(offset,dtype=tf.float64)
        offset = tf.expand_dims(offset, 1)
        offset = tf.broadcast_to(offset, [self.batch_size, self.atom_size])

        proj_dist = tf.reshape(tf.zeros(tf.shape(next_dist)), [-1])
        
        proj_dist = tf.tensor_scatter_nd_add(proj_dist, 
                                             tf.reshape(l + offset, [-1]),
                                             tf.reshape(next_dist * (u - b), [-1]))

        proj_dist = tf.tensor_scatter_nd_add(proj_dist, 
                                             tf.reshape(u + offset, [-1]),
                                             tf.reshape(next_dist * (b - l), [-1]))
            

        dist = self.dqn(state)
        # indexing은 tf.gather_nd 사용하자 <-- 이거 조심. 디버깅 할 때 문제 있으면 이부분 부터 보자.
        index = tf.concat([tf.range(self.batch_size), action], axis=0)
        log_p = tf.math.log(tf.gather_nd(dist, index))
        
        log_p = torch.log(dist[range(self.batch_size), action])
        
        elementwise_loss = -tf.math.reduce_sum(proj_dist * log_p, axis = 1)

        return elementwise_loss
    

env = gym.make('PongDuel-v0')

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
    if abs(bar_y - ball_y) == ball_radius:
        if (bar_x+bar_radius > ball_x and bar_x-bar_radius < ball_x):
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
    
    if (ep_i+1) % 10000 == 0:
        turn ^= 1
