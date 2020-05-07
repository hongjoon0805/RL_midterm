import numpy as np
import tensorflow.compat.v1 as tf
import math

tf.keras.backend.set_floatx('float64')

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
        

    def call(self, x, sample=True):
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
        
        
class OldDuelModel(tf.keras.models.Model):
    def __init__(self, state, action, std=0.1):
        super(OldDuelModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, input_dim=state, kernel_initializer='he_uniform')
        self.fc2 = tf.keras.layers.Dense(128, kernel_initializer='he_uniform')
        
        self.vfc1 = tf.keras.layers.Dense(32, kernel_initializer='he_uniform')
        self.vfc2 = NoisyLinear(32, 1, std_init = std, name = 'vfc2')
        
        self.afc1 = tf.keras.layers.Dense(32, kernel_initializer='he_uniform')
        self.afc2 = NoisyLinear(32, action, std_init = std, name = 'afc2')
        
        self.state = state
        self.action = action
        

    def call(self, x, sample=True):
        feature = tf.nn.relu(self.fc1(tf.cast(x, dtype=tf.float64)))
        feature = tf.cast(tf.nn.relu(self.fc2(feature)), dtype=tf.float64)
        
        value = tf.nn.relu(self.vfc1(feature))
        value = tf.reshape(self.vfc2(value, sample), [-1,1])
        
        advantage = tf.nn.relu(self.afc1(feature))
        advantage = tf.reshape(self.afc2(advantage, sample), [-1, self.action])
        
        output = value + advantage - tf.reshape(tf.math.reduce_mean(advantage, axis=1), [-1,1])
        
        return output
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.vfc2.reset_noise()
        self.afc2.reset_noise()