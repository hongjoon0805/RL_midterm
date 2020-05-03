import numpy as np
import tensorflow.compat.v1 as tf

tf.keras.backend.set_floatx('float64')

class NoisyLinear(tf.keras.layers.Layer):

    def __init__(self, input_dim=32, units=32):
        super(NoisyLinear, self).__init__()
        self.w_mu = self.add_weight(shape=(input_dim, units),initializer='he_uniform',trainable=True, dtype=tf.float64)
        self.w_sigma = self.add_weight(shape=(input_dim, units),
                                       initializer=tf.keras.initializers.Constant(value=0.017),
                                       trainable=True, dtype=tf.float64)
        
        self.b_mu = self.add_weight(shape=(units,),initializer='zeros',trainable=True, dtype=tf.float64)
        self.b_sigma = self.add_weight(shape=(units,),
                                       initializer=tf.keras.initializers.Constant(value=0.017),
                                       trainable=True, dtype=tf.float64)

    
    def call(self, inputs):
        w_eps = tf.keras.backend.random_normal(self.w_mu.shape)
        w = self.w_mu + w_eps * self.w_sigma
        
        b_eps = tf.keras.backend.random_normal(self.b_mu.shape)
        b = self.b_mu + b_eps * self.b_sigma
        
        return tf.matmul(inputs, w) + b

    
class DuelModel(tf.keras.models.Model):
    def __init__(self, state, action, atom):
        super(DuelModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, input_dim=state, kernel_initializer='he_uniform')
        self.fc2 = tf.keras.layers.Dense(128, kernel_initializer='he_uniform')
        
        self.vfc1 = NoisyLinear(128, 32)
        self.vfc2 = NoisyLinear(32, 1*atom)
        
        self.afc1 = NoisyLinear(128, 32)
        self.afc2 = NoisyLinear(32, action*atom)
        
        self.state = state
        self.action = action
        self.atom = atom
        

    def call(self, x):
        feature = tf.nn.relu(self.fc1(x))
        feature = tf.nn.relu(self.fc2(feature))
        
        value = tf.nn.relu(self.vfc1(feature))
        value = tf.reshape(self.vfc2(value), [-1,1,self.atom])
        
        advantage = tf.nn.relu(self.afc1(feature))
        advantage = tf.reshape(self.afc2(advantage), [-1, self.action, self.atom])
        
        output = value + advantage - tf.reshape(tf.math.reduce_mean(advantage, axis=1), [-1,1, self.atom])
        dist = tf.nn.softmax(output, axis=-1)
        dist = tf.clip_by_value(dist,1e-3, 1e8)
        
        return dist