import numpy as np
from collections import deque
from pysgt.StochasticGradientTree import SGTRegressor
import random

class Agent:
    def __init__(self, action_space, gamma = 0.99, learning_rate=0.1,
                 EXPLORATION_MAX = 1, EXPLORATION_MIN = 0.01,
                 EXPLORATION_DECAY = 0.99, batch_size = 64,
                 MEMORY_SIZE = 1000000, epochs=8,
                 upper = [], lower = []):

        self.action_space = action_space

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.exploration_rate = EXPLORATION_MAX
        self.exploration_min = EXPLORATION_MIN
        self.exploration_decay = EXPLORATION_DECAY

        self.batch_size = batch_size

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = [SGTRegressor(epochs=1, bins=8, upper_bounds=upper, lower_bounds=lower, learning_rate=learning_rate) for _ in range(self.action_space)]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)

        q_values = np.array([estimator.predict(state)[0] for estimator in self.model])

        return np.argmax(q_values)

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        for _ in range(self.epochs):
            batch = random.sample(self.memory, self.batch_size)
            current_states = np.array([transition[0][0] for transition in batch])
            current_q_values = np.array([estimator.predict(current_states) for estimator in self.model]).T
            new_states = np.array([transition[3][0] for transition in batch])
            future_q_values = np.array([estimator.predict(new_states) for estimator in self.model]).T

            actions = np.array([transition[1] for transition in batch])
            rewards = np.array([transition[2] for transition in batch])
            dones = np.array([transition[4] for transition in batch])
            
            current_q_values[range(len(current_q_values)), actions] = rewards + self.gamma * np.max(future_q_values, axis=1) * (1 - dones)
        
            [estimator.fit(current_states, current_q_values[:,i]) for i, estimator in enumerate(self.model)]

    def save_model(self, file_name='agent_model.pkl'):
        import pickle
        with open(file_name, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, file_name):
        import pickle
        with open(file_name, 'rb') as f:
            self.model = pickle.load(f)


class TAgent(Agent):
    def __init__(self, action_space, gamma = 0.99, learning_rate=0.1,
                 EXPLORATION_MAX = 1, EXPLORATION_MIN = 0.01,
                 EXPLORATION_DECAY = 0.99, batch_size = 64,
                 MEMORY_SIZE = 1024, epochs=8,
                 upper=[], lower=[]):

        super().__init__(action_space, gamma, learning_rate,
                 EXPLORATION_MAX, EXPLORATION_MIN,
                 EXPLORATION_DECAY, batch_size,
                 MEMORY_SIZE, epochs,
                 upper, lower)

        self.target_model = self.model.copy()

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        for _ in range(self.epochs):
            batch = random.sample(self.memory, self.batch_size)

            current_states = np.array([transition[0][0] for transition in batch])
            current_q_values = np.array([estimator.predict(current_states) for estimator in self.model]).T
            new_states = np.array([transition[3][0] for transition in batch])
            future_q_values = np.array([estimator.predict(new_states) for estimator in self.target_model]).T

            actions = np.array([transition[1] for transition in batch])
            rewards = np.array([transition[2] for transition in batch])
            dones = np.array([transition[4] for transition in batch])
            
            current_q_values[range(len(current_q_values)), actions] = rewards + self.gamma * np.max(future_q_values, axis=1) * (1 - dones)

            [estimator.fit(current_states, current_q_values[:,i]) for i, estimator in enumerate(self.model)]

class DTAgent(Agent):
    def __init__(self, action_space, gamma = 0.99, learning_rate=0.1,
                 EXPLORATION_MAX = 1, EXPLORATION_MIN = 0.01,
                 EXPLORATION_DECAY = 0.99, batch_size = 64,
                 MEMORY_SIZE = 1024, epochs=8,
                 upper=[], lower=[]):

        super().__init__(action_space, gamma, learning_rate,
                 EXPLORATION_MAX, EXPLORATION_MIN,
                 EXPLORATION_DECAY, batch_size,
                 MEMORY_SIZE, epochs,
                 upper, lower)

        self.target_model = self.model.copy()

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        for _ in range(self.epochs):
            batch = random.sample(self.memory, self.batch_size)

            current_states = np.array([transition[0][0] for transition in batch])
            current_q_values = np.array([estimator.predict(current_states) for estimator in self.model]).T
            new_states = np.array([transition[3][0] for transition in batch])
            double_q_values = np.array([estimator.predict(new_states) for estimator in self.model]).T
            future_q_values = np.array([estimator.predict(new_states) for estimator in self.target_model]).T

            actions = np.array([transition[1] for transition in batch])
            rewards = np.array([transition[2] for transition in batch])
            dones = np.array([transition[4] for transition in batch])

            best_next_actions = np.argmax(double_q_values, axis=1)
            
            current_q_values[range(len(current_q_values)), actions] = rewards + self.gamma * future_q_values[range(len(future_q_values)), best_next_actions] * (1 - dones)

            [estimator.fit(current_states, current_q_values[:,i]) for i, estimator in enumerate(self.model)]


class Imitator(Agent):
    def __init__(self, action_space, gamma = 0.99, learning_rate=0.1,
                 EXPLORATION_MAX = 1, EXPLORATION_MIN = 0.01,
                 EXPLORATION_DECAY = 0.99, batch_size = 64,
                 MEMORY_SIZE = 1024, epochs=8,
                 upper=[], lower=[]):

        super().__init__(action_space, gamma, learning_rate,
                 EXPLORATION_MAX, EXPLORATION_MIN,
                 EXPLORATION_DECAY, batch_size,
                 MEMORY_SIZE, epochs,
                 upper, lower)

        import torch
        self.target = torch.load(r'./PolicyDistillation/teacher.pt')
    
    def act(self, state):
        q_values = np.array([estimator.predict(state)[0] for estimator in self.model])
        return np.argmax(q_values)
  
    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        for _ in range(self.epochs):
            batch = random.sample(self.memory, self.batch_size)
            current_states = np.array([transition[0][0] for transition in batch])
            new_states = np.array([transition[3][0] for transition in batch])

            current_q_values = self.target.predict(current_states).numpy()
            future_q_values = self.target.predict(new_states).numpy()
        
            [estimator.fit(current_states, current_q_values[:,i]) for i, estimator in enumerate(self.model)]
            [estimator.fit(new_states, future_q_values[:,i]) for i, estimator in enumerate(self.model)]
