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
from dqn import DQN
import arguments
tf.keras.backend.set_floatx('float64')

def wonjum_daeching(obs):
    x, y = 29/30, 39/40
    old_dir_arr = ['NW', 'W', 'SW', 'SE', 'E', 'NE']
    new_dir_arr = ['NE', 'E', 'SE', 'SW', 'W', 'NW']
    
    obs[1] = y - obs[1]
    obs[3] = y - obs[3]
    
    old_dir = np.argmax(obs[4:])
    obs[4+old_dir] = 0
    new_dir = 5-old_dir
    obs[4+new_dir] = 1
    
    return obs
    
env = gym.make('PongDuel-v0')
args = arguments.get_args()
dqn = DQN(args)
dqn_expert = DQN(args)

import datetime
date = datetime.datetime.now().strftime("%Y-%m-%d")

log_name = '{}_std_{}_lr_{}'.format(
    date,
    args.std,
    args.lr
)

if args.add_1_step_loss:
    log_name += '_add_1_step_loss'
if args.no_tag:
    log_name += '_no_tag'
if args.expert_model != 'ABCD':
    log_name += '_expert'
    dqn.model.load_weights('models/' + args.expert_model)
    dqn.target_model.load_weights('models/' + args.expert_model)
    dqn_expert.model.load_weights('models/' + args.expert_model)


last_100_episode = deque(maxlen=100)
last_100_episode_eval = deque(maxlen=100)

state = env.reset()
frame_cnt = 0
train = False
best_changed = False
best_avg = 7.0
episodes = int(3000)
for ep_i in range(episodes):
    done_n = [False for _ in range(env.n_agents)]
    env.seed(ep_i)
    total_state = np.array(env.reset(), dtype=np.float64)
    state = np.hstack((total_state[0], total_state[1][:2]))
    rewards_cnt = np.array([0,0], dtype=np.float64)
    
    while not all(done_n):
        if args.expert_model == 'ABCD':
            action = [dqn.select_action(state.reshape(1,12)), dqn.select_random_action()]
        else:
            new_obs = wonjum_daeching(state[1])
            action = [dqn.select_action(state[0].reshape(1,10)), dqn_expert.select_action(new_obs.reshape(1,10))]
        total_next_state_n, reward_n, done_n, info = env.step(action)
        next_state = np.hstack((total_next_state_n[0], total_next_state_n[1][:2]))
        
        if all(done_n):
            ball_x, ball_y = next_state[2:4]
            if ball_y > 0.5:
                reward_n[0] += 1
            else:
                reward_n[1] += 1
                
        rewards_cnt += np.array(reward_n)
        
        if args.reward_change:
            reward_n[0] = 3 * reward_n[0]
            reward_n[1] = 3 * reward_n[1]
            
        reward_n[0] = reward_n[0] - 3* reward_n[1]
        reward, done = reward_n[0], done_n[0]
        next_state, reward, done = dqn.pre_process(next_state, reward, done)
        state = next_state

        # if training is ready
        if len(dqn.memory) >= dqn.batch_size:
            train=True
            dqn.update_model()

        frame_cnt += 1
    
    if train:
        dqn.update_target_model()
        dqn.increment_beta(ep_i, episodes)

    last_100_episode.append(rewards_cnt[0])
    
    left_avg = np.mean(last_100_episode)
    dqn.model.save_weights('./models/' + log_name + '.model')
    if left_avg >  best_avg:
        best_avg = left_avg
        best_changed = True
    
    if best_changed:
        dqn.model.save_weights('./models/' + log_name + '_best' + '.model')
        best_changed = False
        print('Best model Saved')
        
    # Evaluation
    eval_reward_cnt = np.array([0,0], dtype=np.float64)
    done_n = [False for _ in range(env.n_agents)]
    total_state = np.array(env.reset(), dtype=np.float64)
    state = np.hstack((total_state[0], total_state[1][:2]))
    while not all(done_n):
        rand_action = np.random.choice(range(3), 1, replace=False)[0]
        action = [dqn.select_action(state.reshape(1,12)), rand_action]
        total_next_state_n, reward_n, done_n, info = env.step(action)
        next_state = np.hstack((total_next_state_n[0], total_next_state_n[1][:2]))

        if all(done_n):
            ball_x, ball_y = next_state[2:4]
            if ball_y > 0.5:
                reward_n[0] += 1
            else:
                reward_n[1] += 1

        eval_reward_cnt += np.array(reward_n)
        state = next_state
    
    last_100_episode_eval.append(eval_reward_cnt[0])
    print('Episode:%d || Score: %d || Avg: %.2f || Eval: %.2f || Max: %.2f'%(ep_i, 
                                                                           rewards_cnt[0],
                                                                           left_avg,
                                                                           np.mean(last_100_episode_eval), 
                                                                           best_avg))
    with open('./log/'+log_name + '.txt', 'a') as f:
        f.write('Episode:%d || Score: %d || Avg: %.2f || Eval: %.2f || Max: %.2f \n'%(ep_i, 
                                                                               rewards_cnt[0],
                                                                               left_avg,
                                                                               np.mean(last_100_episode_eval), 
                                                                               best_avg))
f.close()
