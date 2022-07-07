import sys
sys.path.append(r'./')
from utils.scores import ScoreLogger
from DQN import DQN_double
import time
import torch
import gym
import random

class Experiment:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        
        self.episodes = 501
               
        n_state = self.env.observation_space.shape[0]
        n_action = self.env.action_space.n
        n_hidden = 50
        self.lr = 0.01
        self.model = DQN_double(n_state, n_action, n_hidden, self.lr)

    def q_learning(self, gamma=0.9, 
                epsilon=0.3, eps_decay=0.99,
                replay=False, replay_size=20, 
                double=False, n_update=10, 
                soft=False):
        """Deep Q Learning algorithm using the DQN. """
        memory = []
        episode_i=0
        sum_total_replay_time=0

        params = {
                'method':'Double Q-Learning',
                'env_name': 'CartPole-v1',
                'gamma': gamma,
                'learning_rate': self.lr,
                'eps': epsilon,
                'eps_decay': eps_decay,
                'eps_min': 0.01,
                'batch_size': 'N/A',
                'bins': 'N/A',
                'epochs': 'N/A',
                'category': "Double Q-Learning",
                'prioritized_experience_replay': False,
                'target_model_updates': 0
                }
        score_logger = ScoreLogger(params)

        for episode in range(self.episodes):
            episode_i+=1
            if double and not soft:
                # Update target network every n_update steps
                if episode % n_update == 0:
                    self.model.target_update()
            if double and soft:
                self.model.target_update()
            
            # Reset state
            state = self.env.reset()
            done = False
            total = 0
            step = 0
            
            while not done:
                step += 1
                # Implement greedy search policy to explore the state space
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.model.predict(state)
                    action = torch.argmax(q_values).item()
                
                # Take action and add reward to total
                next_state, reward, done, _ = self.env.step(action)
                
                # Update total and memory
                total += reward
                memory.append((state, action, next_state, reward, done))
                q_values = self.model.predict(state).tolist()
                
                if done:
                    if not replay:
                        q_values[action] = reward
                        # Update network weights
                        self.model.update(state, q_values)
                    score_logger.add_score(step, episode)
                    break

                if replay:
                    t0=time.time()
                    # Update network weights using replay memory
                    self.model.replay(memory, replay_size, gamma)
                    t1=time.time()
                    sum_total_replay_time+=(t1-t0)
                else: 
                    # Update network weights using the last step only
                    q_values_next = self.model.predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                    self.model.update(state, q_values)

                state = next_state
            
            # Update epsilon
            epsilon = max(epsilon * eps_decay, 0.01)


if __name__ == '__main__':
    exp = Experiment('CartPole-v1')
    exp.q_learning(gamma=.99, epsilon=1, replay=True, double=True, n_update=10)