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
import arguments
tf.keras.backend.set_floatx('float64')
    
class DQN:
    def __init__(self, args, mode, random = False):
        
        self.mode_idx = 0 if mode == 'l' else 1
        self.args = args
        self.state_size = 10
        self.action_size = 3
        
        self.memory_size  = args.memory_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        
        # PER parameters
        self.alpha = args.alpha
        self.beta = args.beta
        self.prior_eps = args.prior_eps
        
        # Categorical DQN parameters
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.atom_size = args.atom_size
        self.support = tf.cast(tf.linspace(self.v_min, self.v_max, self.atom_size), dtype=tf.float64)
        
        # N-step Learning
        self.n_step = args.n_step
        
        # Noisy Network
        self.std = args.std
        
        # PER
        # memory for 1-step Learning
        self.memory = PrioritizedReplayBuffer(self.state_size, self.memory_size, self.batch_size, alpha=self.alpha)
        
        # memory for N-step Learning
        self.memory_n = ReplayBuffer(self.state_size, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma)
        
        self.model = DuelModel(self.state_size, self.action_size, self.atom_size, std = self.std)
        self.target_model = DuelModel(self.state_size, self.action_size, self.atom_size, std = self.std)
        
        self.update_target_model()
        
        self.transition = list()
        
        self.frame_cnt = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = args.lr)

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
            
            if self.args.add_1_step_loss:
                # 1-step loss
                elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
            
            # n-step loss
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_n_loss
            
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
        reward = samples["rews"].reshape(-1, 1)
        done = samples["done"].reshape(-1, 1)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        # Double DQN
        q = tf.math.reduce_sum(self.model(next_state) * self.support, axis=2)
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
        
        l_mask = tf.logical_and((u > 0), (l == u))
        u_mask = tf.logical_and((l < (self.atom_size - 1)), (l == u))
        l = tf.where(l_mask, l - 1, l)
        u = tf.where(u_mask, u + 1, u)

        offset = tf.linspace(0.0, float((self.batch_size - 1) * (self.atom_size )), self.batch_size)
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

        # indexing은 tf.gather_nd 사용하자 <-- 이거 조심. 디버깅 할 때 문제 있으면 이부분 부터 보자.
        action = tf.convert_to_tensor(action, dtype = tf.int64)
        index = tf.stack([tf.range(self.batch_size, dtype=tf.int64), action], axis=0)
        index = tf.transpose(index)
        log_p = tf.math.log(tf.gather_nd(self.model(state), index))
        elementwise_loss = -tf.math.reduce_sum(proj_dist * log_p, axis = 1)

        return elementwise_loss
    
    def _cal_reward(self, state, reward):
        bar_x, bar_y = state[:2]
        ball_x, ball_y = state[2:4]
        ball_radius = 0.025
        bar_radius = 0.05
        bar_coord = [bar_x-0.05, bar_x-0.025, bar_x, bar_x+0.025, bar_x+0.05]
        
        if abs(bar_y - ball_y) == ball_radius:
            if abs(bar_x - ball_x) <= bar_radius:
                reward = 0.1
        
        if bar_y==ball_y:
            reward = -3
        
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
        
        dist = self.model(state)
        
        actions = tf.math.reduce_sum(dist * self.support, axis=2)
        selected_action = np.argmax(actions)
        
        self.transition = [state, selected_action]
        
        return selected_action

env = gym.make('PongDuel-v0')
args = arguments.get_args()
dqn = DQN(args, 'l', False)

import datetime
date = datetime.datetime.now().strftime("%Y-%m-%d")

log_name = '{}_std_{}_lr_{}'.format(
    date,
    args.std,
    args.lr
)

if args.add_1_step_loss:
    log_name += 'add_1_step_loss'

f = open('./log/'+log_name + '.txt', 'w')

turn = 0

last_100_episode = deque(maxlen=100)
last_100_episode_eval = deque(maxlen=100)

state = env.reset()
frame_cnt = 0
train = False
best_changed = False
best_avg = 0
episodes = int(3000)
for ep_i in range(episodes):
    done_n = [False for _ in range(env.n_agents)]
    env.seed(ep_i)
    state = np.array(env.reset())
    rewards_cnt = np.array([0,0], dtype=np.float64)
    
    while not all(done_n): 
        rand_action = np.random.choice(range(3), 1, replace=False)[0]
        action = [dqn.select_action(state[0].reshape(1,10)), rand_action]
        next_state_n, reward_n, done_n, info = env.step(action)
        next_state_n = np.array(next_state_n, dtype=np.float64)
        
        if all(done_n):
            ball_x, ball_y = next_state_n[0][2:4]
            if ball_y > 0.5:
                reward_n[0] += 1
            else:
                reward_n[1] += 1
                
        rewards_cnt += np.array(reward_n)
        
        next_state, reward, done = next_state_n[0], reward_n[0], done_n[0]
        next_state, reward, done = dqn.pre_process(next_state, reward, done)
        state = next_state_n

        # if training is ready
        if len(dqn.memory) >= int(5e4):
            train=True
            dqn.update_model()

        frame_cnt += 1
    
    if train:
        dqn.update_target_model()
        dqn.increment_beta(ep_i, episodes)

    last_100_episode.append(rewards_cnt[0])
    
    left_avg = np.mean(last_100_episode)
    
    if left_avg >  best_avg:
        best_avg = left_avg
        best_changed = True
    
    if best_changed:
        dqn.model.save_weights('./models/' + log_name + '.model')
        best_changed = False
        print('Model Saved')
        
    # Evaluation
    eval_reward_cnt = np.array([0,0], dtype=np.float64)
    done_n = [False for _ in range(env.n_agents)]
    state = np.array(env.reset())
    while not all(done_n):
        rand_action = np.random.choice(range(3), 1, replace=False)[0]
        action = [dqn.select_action(state[0].reshape(1,10)), rand_action]
        next_state_n, reward_n, done_n, info = env.step(action)
        next_state_n = np.array(next_state_n, dtype=np.float64)

        if all(done_n):
            ball_x, ball_y = next_state_n[0][2:4]
            if ball_y > 0.5:
                reward_n[0] += 1
            else:
                reward_n[1] += 1

        eval_reward_cnt += np.array(reward_n)
        state = next_state_n
    
    last_100_episode_eval.append(eval_reward_cnt[0])
    print('Episode:%d || Score: %d || Avg: %.2f || Eval: %.2f || Max: %.2f'%(ep_i, 
                                                                           rewards_cnt[0],
                                                                           left_avg,
                                                                           np.mean(last_100_episode_eval), 
                                                                           best_avg))
    
    f.write('Episode:%d || Score: %d || Avg: %.2f || Eval: %d || Max: %.2f \n'%(ep_i, 
                                                                           rewards_cnt[0],
                                                                           left_avg,
                                                                           np.mean(last_100_episode_eval), 
                                                                           best_avg))
f.close()
