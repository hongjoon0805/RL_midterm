import random
import os

import numpy as np
import tensorflow.compat.v1 as tf
import math
tf.keras.backend.set_floatx('float64')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class NoisyLinear(tf.keras.layers.Layer):

    def __init__(self, in_features=32, out_features=32, std_init = 0.2, name='noisy'):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        mu_range = 1 / math.sqrt(in_features)
        
        self.w_mu = self.add_weight(shape=(in_features, out_features),
                                    initializer=tf.random_uniform_initializer(-mu_range,mu_range),
                                    trainable=True, dtype=tf.float64, name=name+'w_mu')
        self.w_sigma = self.add_weight(shape=(in_features, out_features),
                                       initializer=tf.keras.initializers.Constant(value=std_init / math.sqrt(in_features)),
                                       trainable=True, dtype=tf.float64, name=name+'w_sigma')
        
        self.b_mu = self.add_weight(shape=(out_features,),
                                    initializer=tf.random_uniform_initializer(-mu_range,mu_range),
                                    trainable=True, dtype=tf.float64, name=name+'b_mu')
        self.b_sigma = self.add_weight(shape=(out_features,),
                                       initializer=tf.keras.initializers.Constant(value=std_init / math.sqrt(out_features)),
                                       trainable=True, dtype=tf.float64, name=name+'b_sigma')

        
        self.reset_noise()
        
    def reset_noise(self):
        """Make new noise."""
        epsilon_in = tf.reshape(self.scale_noise(self.in_features), [self.in_features, 1])
        epsilon_out = tf.reshape(self.scale_noise(self.out_features), [1, self.out_features])

        # outer product
        self.w_eps = epsilon_in * epsilon_out
        self.b_eps = tf.reshape(epsilon_out, [self.out_features])
    
    def call(self, inputs, sample=False):
        if sample:
            w = self.w_mu + self.w_eps * self.w_sigma
            b = self.b_mu + self.b_eps * self.b_sigma
        else:
            w = self.w_mu
            b = self.b_mu
        
        return tf.matmul(inputs, w) + b
        
    @staticmethod
    def scale_noise(size: int):
        """Set scale to make noise (factorized gaussian noise)."""
        x = tf.random.normal([size], dtype = tf.float64)
        
        return tf.math.sign(x) * tf.math.sqrt(tf.math.abs(x))

class DuelModel(tf.keras.models.Model):
    def __init__(self, state, action, atom, std):
        super(DuelModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, input_dim=state, kernel_initializer='he_uniform')
        self.fc2 = tf.keras.layers.Dense(128, kernel_initializer='he_uniform')
        
        self.vfc1 = tf.keras.layers.Dense(32, kernel_initializer='he_uniform')
        self.vfc2 = NoisyLinear(32, 1*atom, std_init = std, name = 'vfc2')
        
        self.afc1 = tf.keras.layers.Dense(32, kernel_initializer='he_uniform')
        self.afc2 = NoisyLinear(32, action*atom, std_init = std, name = 'afc2')
        
        self.state = state
        self.action = action
        self.atom = atom
        

    def call(self, x, sample=False):
        feature = tf.nn.relu(self.fc1(x))
        feature = tf.cast(tf.nn.relu(self.fc2(feature)), dtype=tf.float64)
        
        value = tf.nn.relu(self.vfc1(feature))
        value = tf.reshape(self.vfc2(value, sample), [-1,1,self.atom])
        
        advantage = tf.nn.relu(self.afc1(feature))
        advantage = tf.reshape(self.afc2(advantage, sample), [-1, self.action, self.atom])
        
        output = value + advantage - tf.reshape(tf.math.reduce_mean(advantage, axis=1), [-1,1, self.atom])
        dist = tf.nn.softmax(output, axis=-1)
#         dist = tf.clip_by_value(dist,1e-3, 1e8)
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.vfc2.reset_noise()
        self.afc2.reset_noise()


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

def start(mode, to_recv, to_send):
    model = DuelModel(10, 3, 21, 0.025)
    model.load_weights('models/2020-05-06_std_0.025_lr_0.000125add_1_step_loss.model')
    support = tf.cast(tf.linspace(-10.0, 10.0, 21), dtype=tf.float64)
    idx = 0 if mode == 'left' else 1
    while True: 
        obs = to_recv.get()
        if obs == None:
            break
#         obs = np.array(obs[idx], dtype=np.float64)
        obs = np.array(obs[idx])
        if mode == 'right':
            obs = wonjum_daeching(obs)
        
        actions = tf.math.reduce_sum(model(obs.reshape((1,10))) * support, axis=2)
        action = np.argmax(actions)
        
        to_send.put(action)