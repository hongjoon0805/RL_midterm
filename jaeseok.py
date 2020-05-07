import random
import os
import tensorflow.compat.v1 as tf
import numpy as np
import math
import h5py
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class NoisyLinear(tf.keras.layers.Layer):
    def __init__(self, in_features=32, out_features=32, std_init = 0.2,name=''):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        mu_range = 1 / math.sqrt(in_features)
        
        self.w_mu = self.add_weight(shape=(in_features, out_features),
                                    initializer=tf.random_uniform_initializer(-mu_range,mu_range),
                                    trainable=True,name=name+'_wmu')#, dtype=tf.float64)
        
        self.w_sigma = self.add_weight(shape=(in_features, out_features),
                                       initializer=tf.keras.initializers.Constant(value=std_init / math.sqrt(in_features)),
                                       trainable=True,name=name+'_wsi')#, dtype=tf.float64)
        
        self.b_mu = self.add_weight(shape=(out_features,),
                                    initializer=tf.random_uniform_initializer(-mu_range,mu_range),
                                    trainable=True,name=name+'_bm')#, dtype=tf.float64)
        
        self.b_sigma = self.add_weight(shape=(out_features,),
                                       initializer=tf.keras.initializers.Constant(value=std_init / math.sqrt(out_features)),
                                       trainable=True,name=name+'_bs')#, dtype=tf.float64)
 
        self.reset_noise()
    def reset_noise(self):
        """Make new noise."""
        epsilon_in = tf.reshape(self.scale_noise(self.in_features), [self.in_features, 1])
        epsilon_out = tf.reshape(self.scale_noise(self.out_features), [1, self.out_features])
        # outer product
        self.w_eps = epsilon_in * epsilon_out
        self.b_eps = tf.reshape(epsilon_out, [self.out_features])
    
    def call(self, inputs):
        w_eps = tf.keras.backend.random_normal(self.w_mu.shape)
        w = self.w_mu + w_eps * self.w_sigma
        
        b_eps = tf.keras.backend.random_normal(self.b_mu.shape)
        b = self.b_mu + b_eps * self.b_sigma        
        return tf.matmul(inputs, w) + b
        
    @staticmethod
    def scale_noise(size: int):
        """Set scale to make noise (factorized gaussian noise)."""
        x = tf.random.normal([size], dtype = tf.float64)
        return tf.math.sign(x) * tf.math.sqrt(tf.math.abs(x))

class Duel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions,version):
        super(Duel, self).__init__()
        self.version=version
        
        self.feature1=tf.keras.layers.Dense(128,activation='relu',kernel_initializer='he_uniform')
        self.feature2=tf.keras.layers.Dense(128, kernel_initializer='he_uniform')

        self.advantage_hidden_layers = []
        for i in hidden_units:
            self.advantage_hidden_layers.append(tf.keras.layers.Dense(
                i, activation='relu', kernel_initializer='he_uniform'))
        if 'noisy' in version:
           
           self.noisy_output_advantage=NoisyLinear(32,num_actions,std_init = 0.2,name='a1')
        else:
           self.output_advantage=tf.keras.layers.Dense(num_actions,kernel_initializer='he_uniform')
 
        self.value_hidden_layers = []
        for i in hidden_units:
            self.value_hidden_layers.append(tf.keras.layers.Dense(
                i, activation='relu', kernel_initializer='he_uniform'))

        if 'noisy' in version:
           self.noisy_output_value=NoisyLinear(32,1,std_init = 0.2,name='v1')
        else:
           self.output_value=tf.keras.layers.Dense(1,kernel_initializer='he_uniform')
 
    @tf.function
    def call(self, inputs):
        z = self.feature2(self.feature1(inputs))

        for layer in self.advantage_hidden_layers:
            A = layer(z)
        if 'noisy' in self.version:        
            A=self.noisy_output_advantage(A)
        else:
            A=self.output_advantage(A) 

        for layer in self.value_hidden_layers:
            V = layer(z)
        if 'noisy' in self.version:   
            V=self.noisy_output_value(V)
        else:
            V=self.output_value(V)     
        output = V+A-tf.reshape(tf.math.reduce_mean(A,axis=-1),[-1,1])
        return output

    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_output_value.reset_noise()
        self.noisy_output_advantage.reset_noise()

def _build_network_duel():
     model=Duel(10,[32],3,'duel') 
     return model

def _build_noisy_duel():
     model=Duel(10,[32],3,'noisy') 
     return model

def _build_network_dqn():
     model=tf.keras.models.Sequential()
     model.add(tf.keras.layers.Dense(32, input_dim=10,activation='relu', kernel_initializer='he_uniform'))
     model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
     model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')) 
     model.add(tf.keras.layers.Dense(3,activation='linear', kernel_initializer='he_uniform'))
     return model

def start(mode, to_recv, to_send):
    model=_build_network_duel()
    # model.load_weights(file_name)
    # print(file_name)
    file_name='left_real_new_duel_per_0.995.h5'   
    index=0
    while True: 
        index+=1
        obs = to_recv.get()
        if obs == None:
           break
        if mode=='left':
           input_obs=np.asarray(obs[0])
        else:
           input_obs=np.asarray(obs[1])           
        input_obs=input_obs.flatten()
        input_obs=np.expand_dims(input_obs,axis=0)
        if mode!='left':
            input_obs[:,1]= (29-input_obs[:,1]*30 )/30
            input_obs[:,3]= (29-input_obs[:,3]*30)/30
            if input_obs[:,4]==1:
                input_obs[:,4]=0
                input_obs[:,9]=1
            elif input_obs[:,5]==1:
                input_obs[:,5]=0
                input_obs[:,8]=1 
            elif input_obs[:,6]==1:
                input_obs[:,6]=0
                input_obs[:,7]=1
            elif input_obs[:,7]==1:
                input_obs[:,7]=0
                input_obs[:,6]=1
            elif input_obs[:,8]==1:
                input_obs[:,8]=0
                input_obs[:,5]=1
            elif input_obs[:,9]==1:
                input_obs[:,9]=0
                input_obs[:,4]=1
        q_values=model(input_obs)
        if index==1:
           # print(model.summary())
            data=h5py.File(file_name,"r")
#            print(data.keys())
#            print(model.weights)
            model.load_weights(file_name)
            q_values=model(input_obs)
        #print(model.summary())
        action = np.argmax(q_values[0])
        to_send.put(action)

