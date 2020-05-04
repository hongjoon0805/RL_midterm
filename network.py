import numpy as np
import tensorflow.compat.v1 as tf
import math

tf.keras.backend.set_floatx('float64')

class BayesLinear(tf.keras.layers.Layer):

    def __init__(self, in_features, out_features, std_init = 1/128):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        total_var = 2 / in_features
        noise_var = total_var * std_init
        mu_var = total_var - noise_var
        
        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std)-1)
        
        self.w_mu = self.add_weight(shape=(in_features, out_features),
                                    initializer=tf.random_uniform_initializer(-bound,bound),
                                    trainable=True, dtype=tf.float64)
        self.w_rho = self.add_weight(shape=(out_features,),
                                       initializer=tf.keras.initializers.Constant(rho_init),
                                       trainable=True, dtype=tf.float64)
        
        self.b = self.add_weight(shape=(out_features,),
                                    initializer='zeros',
                                    trainable=True, dtype=tf.float64)
        
    
    def call(self, inputs, sample=False):
        if sample:
            eps = tf.keras.backend.random_normal(self.w_mu.shape)
            w_std = tf.math.log1p(tf.math.exp(self.w_rho))
            w = self.w_mu + w_eps * w_std
        else:
            w = self.w_mu
            
        b = self.b
        return tf.matmul(inputs, w) + b
        
# class NoisyLinear(tf.keras.layers.Layer):

#     def __init__(self, in_features=32, out_features=32, std_init = 0.2):
#         super(NoisyLinear, self).__init__()
        
#         self.in_features = in_features
#         self.out_features = out_features
        
#         mu_range = 1 / math.sqrt(in_features)
        
#         self.w_mu = self.add_weight(shape=(in_features, out_features),
#                                     initializer=tf.random_uniform_initializer(-mu_range,mu_range),
#                                     trainable=True, dtype=tf.float64)
#         self.w_sigma = self.add_weight(shape=(in_features, out_features),
#                                        initializer=tf.keras.initializers.Constant(value=std_init / math.sqrt(in_features)),
#                                        trainable=True, dtype=tf.float64)
        
#         self.b_mu = self.add_weight(shape=(out_features,),
#                                     initializer=tf.random_uniform_initializer(-mu_range,mu_range),
#                                     trainable=True, dtype=tf.float64)
#         self.b_sigma = self.add_weight(shape=(out_features,),
#                                        initializer=tf.keras.initializers.Constant(value=std_init / math.sqrt(out_features)),
#                                        trainable=True, dtype=tf.float64)

        
#         self.reset_noise()
        
#     def reset_noise(self):
#         """Make new noise."""
#         epsilon_in = tf.reshape(self.scale_noise(self.in_features), [self.in_features, 1])
#         epsilon_out = tf.reshape(self.scale_noise(self.out_features), [1, self.out_features])

#         # outer product
#         self.w_eps = epsilon_in * epsilon_out
#         self.b_eps = tf.reshape(epsilon_out, [self.out_features])
    
#     def call(self, inputs):
#         w = self.w_mu + self.w_eps * self.w_sigma
#         b = self.b_mu + self.b_eps * self.b_sigma
        
#         return tf.matmul(inputs, w) + b
        
#     @staticmethod
#     def scale_noise(size: int):
#         """Set scale to make noise (factorized gaussian noise)."""
#         x = tf.random.normal([size], dtype = tf.float64)
        
#         return tf.math.sign(x) * tf.math.sqrt(tf.math.abs(x))


class NoisyLinear(tf.keras.layers.Layer):

    def __init__(self, input_dim=32, units=32, std_init = 0.2):
        super(NoisyLinear, self).__init__()
        self.w_mu = self.add_weight(shape=(input_dim, units),initializer='he_uniform',trainable=True)
        self.w_sigma = self.add_weight(shape=(input_dim, units),
                                       initializer=tf.keras.initializers.Constant(value=0.017),
                                       trainable=True)
        
        self.b_mu = self.add_weight(shape=(units,),initializer='zeros',trainable=True)
        self.b_sigma = self.add_weight(shape=(units,),
                                       initializer=tf.keras.initializers.Constant(value=0.017),
                                       trainable=True)

    
    def call(self, inputs):
        w_eps = tf.keras.backend.random_normal(self.w_mu.shape)
        w = self.w_mu + w_eps * self.w_sigma
        
        b_eps = tf.keras.backend.random_normal(self.b_mu.shape)
        b = self.b_mu + b_eps * self.b_sigma
        
        return tf.matmul(inputs, w) + b
    
    def reset_noise(self):
        pass

    
class DuelModel(tf.keras.models.Model):
    def __init__(self, state, action, atom, std):
        super(DuelModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, input_dim=state, kernel_initializer='he_uniform')
        self.fc2 = tf.keras.layers.Dense(128, kernel_initializer='he_uniform')
        
        self.vfc1 = tf.keras.layers.Dense(32, kernel_initializer='he_uniform')
        self.vfc2 = NoisyLinear(32, 1*atom, std_init = std)
        
        self.afc1 = tf.keras.layers.Dense(32, kernel_initializer='he_uniform')
        self.afc2 = NoisyLinear(32, action*atom, std_init = std)
        
        self.state = state
        self.action = action
        self.atom = atom
        

    def call(self, x, sample=False):
        feature = tf.nn.relu(self.fc1(x))
        feature = tf.cast(tf.nn.relu(self.fc2(feature)), dtype=tf.float64)
        
        value = tf.nn.relu(self.vfc1(feature))
        value = tf.reshape(self.vfc2(value), [-1,1,self.atom])
        
        advantage = tf.nn.relu(self.afc1(feature))
        advantage = tf.reshape(self.afc2(advantage), [-1, self.action, self.atom])
        
        output = value + advantage - tf.reshape(tf.math.reduce_mean(advantage, axis=1), [-1,1, self.atom])
        dist = tf.nn.softmax(output, axis=-1)
        dist = tf.clip_by_value(dist,1e-3, 1e8)
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.vfc2.reset_noise()
        self.afc2.reset_noise()
        
        
class OldDuelModel(tf.keras.models.Model):
    def __init__(self, state, action, std=0.2):
        super(OldDuelModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, input_dim=state, kernel_initializer='he_uniform')
        self.fc2 = tf.keras.layers.Dense(128, kernel_initializer='he_uniform')
        
        self.vfc1 = NoisyLinear(128,32)
        self.vfc2 = NoisyLinear(32, 1, std_init = std)
        
        self.afc1 = NoisyLinear(128,32)
        self.afc2 = NoisyLinear(32, action, std_init = std)
        
        self.state = state
        self.action = action
        

    def call(self, x, sample=False):
        feature = tf.nn.relu(self.fc1(x))
        feature = tf.cast(tf.nn.relu(self.fc2(feature)), dtype=tf.float64)
        
        value = tf.nn.relu(self.vfc1(feature))
        value = tf.reshape(self.vfc2(value), [-1,1])
        
        advantage = tf.nn.relu(self.afc1(feature))
        advantage = tf.reshape(self.afc2(advantage), [-1, self.action])
        
        output = value + advantage - tf.reshape(tf.math.reduce_mean(advantage, axis=1), [-1,1])
        
        return output
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.vfc2.reset_noise()
        self.afc2.reset_noise()