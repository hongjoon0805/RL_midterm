import sys
import numpy as np
import tensorflow.compat.v1 as tf
import random
import gym
from collections import deque


class DQN:
    def __init__(self, env, multistep=False, random = False):

        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False
        self.random = random
        self.n_steps = 10            # Multistep(n-step) 구현 시 n 값, 수정 가능
        if not multistep:
            self.n_steps = 1
        
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.mini_batch_size = 64
        self.gamma = 1.00
        
        self.model = self._build_network()
        self.target_model = self._build_network()
        
        self.update_target_model()
        
        buffer_size = 10000
        
        self.buffer_dict = {'state':deque(maxlen=int(buffer_size)), 
                       'action':deque(maxlen=int(buffer_size)), 
                       'reward':deque(maxlen=int(buffer_size)),
                       'next_state':deque(maxlen=int(buffer_size)),
                       'done':deque(maxlen=int(buffer_size))}
        self.frame_cnt = 0
        self.optimizer = tf.keras.optimizers.Adam()

    def _build_network(self):
        
        # Target 네트워크와 Local 네트워크를 설정
        unitN = 32
        
#         with tf.device('/cpu:0'):

#             model = tf.keras.models.Sequential()
#             model.add(tf.keras.layers.Dense(unitN, input_dim=self.state_size, activation='relu',
#                             kernel_initializer='he_uniform'))
#             model.add(tf.keras.layers.Dense(unitN, activation='relu',
#                             kernel_initializer='he_uniform'))
#             model.add(tf.keras.layers.Dense(self.action_size, activation='linear',
#                             kernel_initializer='he_uniform'))
        
#         adam = tf.keras.optimizers.Adam()
#         model.compile(loss='mse', optimizer=adam)
        
        with tf.device('/cpu:0'):
            inputs = tf.keras.layers.Input(shape = (self.state_size,))
            layer = tf.keras.layers.Dense(unitN, kernel_initializer = 'he_normal')(inputs)
            features = tf.keras.layers.Activation('relu')(layer)
            
            # Advantage layer
            layerA = tf.keras.layers.Dense(unitN, kernel_initializer = 'he_normal')(features)
            layerA = tf.keras.layers.Activation('relu')(layerA)
            layerA = tf.keras.layers.Dense(self.action_size, kernel_initializer = 'he_normal')(layerA)
            layerA = tf.keras.layers.Activation('relu')(layerA)
            
            # Value layer
            layerV = tf.keras.layers.Dense(unitN, kernel_initializer = 'he_normal')(features)
            layerV = tf.keras.layers.Activation('relu')(layerV)
            layerV = tf.keras.layers.Dense(1, kernel_initializer = 'he_normal')(layerV)
            layerV = tf.keras.layers.Activation('relu')(layerV)
            
            output = layerV + layerA - tf.reshape(tf.math.reduce_mean(layerA, axis=1), [-1, 1])
            
            model = tf.keras.models.Model(inputs = inputs, outputs = output)
            
        return model

    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환
        
        if self.random:
            return np.random.choice(range(self.action_size), 1, replace=False)
        
        num = np.random.random_sample()
        if num > self.eps:
            actions = self.model(state)
            action = np.argmax(actions)
        else:
            action = np.random.choice(range(self.action_size), 1, replace=False)
        
        action = int(action)
        
        return action
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def train_minibatch(self, X, Y):
        # mini batch를 받아 policy를 update
        
        with tf.GradientTape() as tape:
            loss = tf.math.reduce_mean(tf.square(Y - self.model(X)))
        model_vars = self.model.trainable_variables
        grad = tape.gradient(loss, model_vars)
        self.optimizer.apply_gradients(zip(grad, model_vars))
        
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
        bsize = self.mini_batch_size
        batch_idx = np.random.choice(list(range(len(buffer_dict['state'])-(self.n_steps-1))), bsize, replace=False)
            
        states = np.array(buffer_dict['state'])[batch_idx]
        actions = np.array(buffer_dict['action'])[batch_idx]
        if self.multistep:
            rewards_arr = np.zeros((bsize,self.n_steps))
            gamma_arr = np.ones((self.n_steps,))
            for i in range(self.n_steps):
                rewards_arr[:,i] = np.array(buffer_dict['reward'])[batch_idx+i]
                if i > 0:
                    gamma_arr[i] = self.gamma * gamma_arr[i-1]
            
            rewards = (rewards_arr * gamma_arr).sum(axis=1)
            
        else:
            rewards = np.array(buffer_dict['reward'])[batch_idx]
        next_states = np.array(buffer_dict['next_state'])[batch_idx + (self.n_steps-1)]
        dones = np.array(buffer_dict['done'])[batch_idx]
        
        action_target = self.model(states).numpy()
        action_max = np.argmax(action_target, axis=1)
        
        q_value = self.target_model(next_states).numpy()[np.arange(bsize), action_max]
        q_target_out = rewards + (self.gamma ** self.n_steps) * q_value
        target = q_target_out * (1-dones) + rewards * (dones)
        action_target[np.arange(bsize), actions] = target
        
        X, Y = states, action_target
        
#         action_target = self.model.predict(states, batch_size = bsize)
        
#         q_target_out = rewards + (self.gamma ** self.n_steps) * self.target_model.predict(next_states, batch_size = bsize, verbose=0).max(axis=1)
#         target = q_target_out * (1-dones) + rewards * (dones)
#         action_target[np.arange(bsize),actions] = target
        
#         X, Y = states, action_target
        
        return X, Y
    

env = gym.make('CartPole-v1')
dqn = DQN(env, True, False)


last_100_episode_step_count = deque(maxlen=100)
avg_step_count_list = []
frame_cnt = 0
for ep_i in range(10000):
    done = False
    state = env.reset()
    step_count = 0
    
    
    while not done: 
        action = dqn.predict(state.reshape(1,dqn.state_size))
        next_state, reward, done, _ = env.step(action)
        
        if step_count == 499:
            reward += 100
        
        if done and step_count < 499:
            reward -= 100
        
        dqn.update_buffer(state, action, reward, next_state, done)
        if frame_cnt > 1000:
            X, Y = dqn.make_minibatch(dqn.buffer_dict)
            dqn.train_minibatch(X,Y)
            
        frame_cnt += 1
        step_count += 1
        state = next_state
        
    dqn.update_epsilon()
    dqn.update_target_model()
    
    
    last_100_episode_step_count.append(step_count)
    
    if np.mean(last_100_episode_step_count) >= 475:
        break
            
    # 최근 100개의 에피소드 평균 step 횟수를 저장 (이 부분은 수정하지 마세요)
    avg_step_count = np.mean(last_100_episode_step_count)
    print("[Episode {:>5}]  episode step_count: {:>5} avg step_count: {}".format(ep_i, step_count, avg_step_count))

    avg_step_count_list.append(avg_step_count)