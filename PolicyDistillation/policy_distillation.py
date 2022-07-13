import sys
sys.path.append(r'./')
import numpy as np
import time
import math
from utils.scores import ScoreLogger
from rl_models.SGTAgent import Imitator
import gym

class Experiment:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.category = 'CartPole_Gym_LearningRate_Distillation'
    
    def policy_learning(self, evaluate=True):        
        observation_space = self.env.observation_space.shape[0]
        action_space = self.env.action_space.n
        
        upper = np.array([self.env.observation_space.high[0], 3.5, self.env.observation_space.high[2], math.radians(175)])
        lower = np.array([self.env.observation_space.low[0], -3.5, self.env.observation_space.low[2], -math.radians(175)])
        agent = Imitator(action_space, upper=upper, lower=lower)
        
        params = {
            'method': 'Teacher-student',
            'env_name': self.env_name,
            'gamma': 'N/A',
            'learning_rate':'N/A',
            'eps': 'N/A',
            'eps_decay': 'N/A',
            'eps_min': 'N/A',
            'batch_size': agent.batch_size,
            'bins': agent.model[0].getBins(),
            'epochs': agent.epochs,
            'category': self.category,
            'prioritized_experience_replay': False,
            'target_model_updates': 0
            }
        score_logger = ScoreLogger(params)

        start = time.process_time()
        for episode in range(1, 101):
            state = self.env.reset()
            state = np.reshape(state, [1, observation_space])
            step = 0
            while True:
                step += 1
                #self.env.render()
                action = agent.act(state)
                state_next, reward, done, _ = self.env.step(action)
                state_next = np.reshape(state_next, [1, observation_space])
                agent.remember(state, action, reward, state_next, done)
                state = state_next
                agent.experience_replay()
                if done:
                    print('-------------------------------')
                    print("Episode: " + str(episode) + ", exploration: " + str(agent.exploration_rate) + ", score: " + str(step))
                    print("Learning rate: ", agent.model[0].lr)
                    score_logger.add_score(step, episode)
                    if agent.model[0]._isFit:
                        print('tree sizes: ', [tree.get_depth() for tree in agent.model])
                    break

        agent.save_model(score_logger.folder_path+'SGTAgent.pkl')

        if evaluate:
            self.agent_evaluate(agent, score_logger)

        self.env.close()
        print('Time taken: {} minutes'.format((time.process_time() - start)/60))
        print('Total nodes: ',[estimator.get_total_nodes() for estimator in agent.model])

    def agent_evaluate(self, agent, score_logger):
        for episode in range(1, 11):
                state = self.env.reset()
                state = np.reshape(state, [1, self.env.observation_space.shape[0]])
                step = 0
                while True:
                    step += 1
                    #env.render()
                    action = np.argmax([estimator.predict(state)[0] for estimator in agent.model])
                    print(f'Took action {action}')
                    for i, estimator in enumerate(agent.model):
                        print(f'SGT {i+1}:\nNode path: {estimator.tree.explain(state[0])}')
                    state_next, _, done, _ = self.env.step(action)
                    print(f'Landed at state: {state_next}\n')
                    state_next = np.reshape(state_next, [1, self.env.observation_space.shape[0]])
                    state = state_next
                    if done:
                        print('-------------------------------')
                        print("Evaluation Episode: " + str(episode) + ", score: " + str(step))
                        score_logger.add_score(step, episode)
                        if agent.model[0]._isFit:
                            print('tree sizes: ', [tree.get_depth() for tree in agent.model])
                        break

if __name__ == "__main__":
    from DeepQLearning.DQN import DQN_double
    exp = Experiment('CartPole-v1')
    exp.policy_learning()       