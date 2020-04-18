import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import random

class DQLAgent():
    
    def __init__(self, env):
        
        #hyperparameter
        self.state_size = env.observation_space.shape[0]                    #Input Dim
        self.action_size = env.action_space.n                               #Output Dim
       
        self.gamma = 0.95 #Whether to focus on future or present rewards
        self.learning_rate = 0.001
        
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 1000)
        self.model = self.build_model()
    
    def build_model(self):
        
        model = Sequential()
        model.add(Dense(48,input_dim =self.state_size))
        model.add(Activation('tanh'))
        
        model.add(Dense(self.action_size))
        model.add(Activation('linear'))
        
        model.compile(loss ="mse", optimizer = Adam(lr = self.learning_rate))
        
        return model
        
    
    def remember(self, state, action, reward, next_state,done):
        
        #Store information related to the environment!!!
        self.memory.append((state, action, reward, next_state,done))  
        
    
    def act(self,state):
        
        #Explore or Exploit
        #Acts according to the information of the STATE.
        
        if random.uniform(0,1) <= self.epsilon:
            return env.action_space.sample()
        
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
    
    
    def replay(self,batch_size):
        #How much of the memory is used (batch_size) + Training
        
        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory,batch_size)
        
        for state, action, reward, next_state,done in mini_batch:
            if done:
                target = reward
            
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
                
            train_target = self.model.predict(state)
            train_target[0][action] = target    
            self.model.fit(state,train_target, verbose = 0)
        
    
    def adaptive_greedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    print ('object created')
