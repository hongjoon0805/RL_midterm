import sys
import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_




# ## Noisy Layer
# 
# Please see *05.noisy_net.ipynb* for detailed description.
# 

# Keras custom layer로 바꾸자.

class NoisyLinear(tf.keras.layers.Layer):

    def __init__(self, input_dim=32, units=32):
        super(NoisyLinear, self).__init__()
        self.w_mu = self.add_weight(shape=(input_dim, units),initializer='random_normal',trainable=True)
        self.w_sigma = self.add_weight(shape=(input_dim, units),
                                       initializer=tf.keras.initializers.Constant(value=0.014),
                                       trainable=True)
        
        self.b_mu = self.add_weight(shape=(units,),initializer='zeros',trainable=True)
        self.b_sigma = self.add_weight(shape=(units,),
                                       initializer=tf.keras.initializers.Constant(value=0.014),
                                       trainable=True)

    
    def call(self, inputs):
        w_eps = tf.keras.backend.random_normal(self.w_mu.shape)
        w = self.w_mu + w_eps * self.w_sigma
        
        b_eps = tf.keras.backend.random_normal(self.b_mu.shape)
        b = self.b_mu + b_eps * self.b_sigma
        
        return tf.matmul(inputs, w) + b

class Network(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
# ## NoisyNet + DuelingNet + Categorical DQN
# 
# #### NoisyNet + DuelingNet
# 
# NoisyLinear is employed for the last two layers of advantage and value layers. The noise should be reset at evey update step.
# 
# #### DuelingNet + Categorical DQN
# 
# The dueling network architecture is adapted for use with return distributions. The network has a shared representation, which is then fed into a value stream with atom_size outputs, and into an advantage stream with atom_size × out_dim outputs. For each atom, the value and advantage streams are aggregated, as in dueling DQN, and then passed through a softmax layer to obtain the normalized parametric distributions used to estimate the returns’ distributions.
# 
# ```
#         advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
#         value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
#         q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
#         
#         dist = F.softmax(q_atoms, dim=-1)
# ```


# ## Rainbow Agent
# 
# Here is a summary of DQNAgent class.
# 
# | Method           | Note                                                 |
# | ---              | ---                                                  |
# |select_action     | select an action from the input state.               |
# |step              | take an action and return the response of the env.   |
# |compute_dqn_loss  | return dqn loss.                                     |
# |update_model      | update the model by gradient descent.                |
# |target_hard_update| hard update from the local model to the target model.|
# |train             | train the agent during num_frames.                   |
# |test              | test the agent (1 episode).                          |
# |plot              | plot the training progresses.                        |
# 
# #### Categorical DQN + Double DQN
# 
# The idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation. Here, we use `self.dqn` instead of `self.dqn_target` to obtain the target actions.
# 
# ```
#         # Categorical DQN + Double DQN
#         # target_dqn is used when we don't employ double DQN
#         next_action = self.dqn(next_state).argmax(1)
#         next_dist = self.dqn_target.dist(next_state)
#         next_dist = next_dist[range(self.batch_size), next_action]
# ```



class DQNAgent:

    def __init__(
        self, 
        memory_size: int = 1e6,
        batch_size: int = 32,
        target_update: int = 100,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
    ):

        obs_dim = 12
        action_dim = 2
        
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha
        )
        
        # memory for N-step Learning
        self.use_n_step = True 
        self.n_step = n_step
        self.memory_n = ReplayBuffer(
            obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
        )
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = tf.linspace(self.v_min, self.v_max, self.atom_size)

        # networks: dqn, dqn_target
        self.dqn = self._build_network(obs_dim, action_dim, self.atom_size, self.support)
        self.dqn_target = self._build_network(obs_dim, action_dim, self.atom_size, self.support)
        
        self.update_target_model()
        
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False
    
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

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
            value, advantage = self.dqn(state)
            
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
        elementwise_loss += elementwise_loss_n_loss

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        model_weights = self.dqn.trainable_variables
        grad = tape.gradient(loss, model_weights)
        self.optimizer.apply_gradients(zip(grad, model_weights))
        
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()
        
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            
            # NoisyNet: removed decrease of epsilon
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses)
                
        self.env.close()
                
    def test(self) -> None:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        while not done:
            self.env.render()
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()

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

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = tf.clip_by_value(self.v_min, self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = tf.cast(tf.math.floor(b), dtype=tf.long)
            u = tf.cast(tf.math.ceil(b), dtype=tf.long)
            
            offset = tf.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size)
            offset = tf.cast(offset,dtype=tf.long)
            offset = tf.expand_dims(offset, 1)
            offset = tf.broadcast_to(offset, [self.batch_size, self.atom_size])
            
            
            proj_dist = tf.zeros(tf.shape(next_dixt))
            proj_dist = tf.reshape(proj_dist, [-1])
            proj_dist = tf.tensor_scatter_nd_add(proj_dist, 
                                                 tf.reshape(l + offset, [-1]),
                                                 tf.reshape(next_dist * (tf.cast(u, ) - b))
                                                 (next_dist * (u.float() - b)).reshape(-1))
            
            proj_dist = tf.tensor_scatter_nd_add(proj_dist, 
                                                 tf.reshape(l + offset, [-1]),
                                                 tf.reshape(next_dist * (tf.cast(u, ) - b))
                                                 (next_dist * (u.float() - b)).reshape(-1))
            
            
            proj_dist = tf.tensor_scatter_nd_add(proj_dist, (l + offset).reshape(-1),(next_dist * (u.float() - b)).reshape(-1))
            
            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.reshape(-1).index_add_(
                0, (l + offset).reshape(-1), (next_dist * (u.float() - b)).reshape(-1)
            )
            proj_dist.reshape(-1).index_add_(
                0, (u + offset).reshape(-1), (next_dist * (b - l.float())).reshape(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                