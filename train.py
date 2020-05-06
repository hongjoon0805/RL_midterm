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
    
# rainbow 구현 모두 끝나면 reward 체계를 바꿔보자. bar에 닿는 횟수가 매우 적으니, bar에 닿을 때 마다 reward를 100정도 얻을 수 있게 해보자.
class DQN:
    def __init__(self, mode, random = False):
        
        self.mode_idx = 0 if mode == 'l' else 1
            
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
        self.support = tf.cast(tf.linspace(self.v_min, self.v_max, self.atom_size), dtype=tf.float64)
        
        # N-step Learning
        self.n_step = 3
        
        # Noisy Network
        self.std = 0.2
        
        # PER
        # memory for 1-step Learning
        self.memory = PrioritizedReplayBuffer(self.state_size, self.memory_size, self.batch_size, alpha=self.alpha)
        
        # memory for N-step Learning
        self.memory_n = ReplayBuffer(self.state_size, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma)
        
        self.model = OldDuelModel(self.state_size, self.action_size, std = self.std)
        self.target_model = OldDuelModel(self.state_size, self.action_size, std = self.std)
        
        self.update_target_model()
        
        self.transition = list()
        
        self.frame_cnt = 0
        self.optimizer = tf.keras.optimizers.Adam()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def increment_beta(self, episode_idx, total_episode):
        # PER: increase beta
        fraction = min(episode_idx / total_episode, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)
    
    def update_model(self):
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = samples["weights"]
        indices = samples["indices"]
        
        # N-step Learning loss
        
        with tf.GradientTape() as tape:
            
            # n-step loss
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss = self._compute_dqn_loss(samples, gamma)
            
            # PER: importance sampling before average
            
            loss = tf.math.reduce_mean(elementwise_loss * weights)
        
        model_weights = self.model.trainable_variables
        grad = tape.gradient(loss, model_weights)
        self.optimizer.apply_gradients(zip(grad, model_weights))
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        self.model.reset_noise()
        self.target_model.reset_noise()
    
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float):
        """Return categorical dqn loss."""
        
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"]
        done = samples["done"]

        # Double DQN
        next_action = tf.math.argmax(self.model(next_state), axis=1)
        index = tf.stack([tf.range(self.batch_size, dtype=tf.int64), next_action], axis=0)
        index = tf.transpose(index)
        
        target_q = tf.gather_nd(self.target_model(next_state), index)

        action = tf.convert_to_tensor(action, dtype = tf.int64)
        index = tf.stack([tf.range(self.batch_size, dtype=tf.int64), action], axis=0)
        index = tf.transpose(index)
        
        q = tf.gather_nd(self.model(state), index)
        
        elementwise_loss = tf.math.square(target_q * gamma * (1-done) + reward - q)

        return elementwise_loss
    
    def _cal_reward(self, state, reward):
#         bar_x, bar_y = state[:2]
#         ball_x, ball_y = state[2:4]
#         ball_radius = 0.025
#         bar_radius = 0.05
#         bar_coord = [bar_x-0.05, bar_x-0.025, bar_x, bar_x+0.025, bar_x+0.05]
# #         if abs(bar_y - ball_y) == ball_radius:
# #             if ball_x in bar_coord:
# #                 reward = 5

#         if bar_y==ball_y:
#             reward = -3
        
        return reward
    
    def pre_process(self, next_state: np.ndarray, reward, done) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        reward = self._cal_reward(next_state, reward)
        
        self.transition += [reward, next_state, done]

        # N-step transition
        one_step_transition = self.memory_n.store(*self.transition)
        
        # add a single step transition
        if one_step_transition:
            self.memory.store(*one_step_transition)
        
        return next_state, reward, done
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        
        if self.random:
            return np.random.choice(range(self.action_size), 1, replace=False)[0]
        
        selected_action = np.argmax(self.model(state))
        self.transition = [state, selected_action]
        
        return selected_action

env = gym.make('PongDuel-v0')
dqn = [DQN('l', False), DQN('r', True)]

f = open('log.txt', 'w')

turn = 0

last_100_episode = [deque(maxlen=100), deque(maxlen=100)]

state = env.reset()
frame_cnt = 0
episodes = int(3000)
for ep_i in range(episodes):
    done_n = [False for _ in range(env.n_agents)]
    env.seed(ep_i)
    state = np.array(env.reset())
    rewards_cnt = np.array([0,0], dtype=np.float64)
    
    while not all(done_n): 
        action = [dqn[0].select_action(state[0].reshape(1,10)), dqn[1].select_action(state[1].reshape(1,10))]
        next_state_n, reward_n, done_n, info = env.step(action)
        next_state_n = np.array(next_state_n, dtype=np.float64)
        
        if all(done_n):
            ball_x, ball_y = next_state_n[0][2:4]
            if ball_y > 0.5:
                reward_n[0] += 1
            else:
                reward_n[1] += 1
                
        rewards_cnt += np.array(reward_n)
        
        reward_n[0] = reward_n[0] - reward_n[0]
        
        next_state, reward, done = next_state_n[turn], reward_n[turn], done_n[turn]
        next_state, reward, done = dqn[turn].pre_process(next_state, reward, done)
        state = next_state_n

        # if training is ready
        if len(dqn[turn].memory) >= dqn[turn].batch_size:
            dqn[turn].update_model()

        frame_cnt += 1

    dqn[turn].update_target_model()
    dqn[turn].increment_beta(ep_i, episodes)
    dqn[turn].model.save_weights("PER.model")

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
