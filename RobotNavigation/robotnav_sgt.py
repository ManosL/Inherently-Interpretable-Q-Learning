from ast import Import
import sys
sys.path.append(r'./')
import numpy as np
import time
from tqdm.auto import tqdm
from rl_models.SGTAgent import Agent
from utils.scores import ScoreLogger
try:
    from RobotNavigation.RobotNavigation import RobotNavEnv
except ImportError:
    from RobotNavigation import RobotNavEnv

class Experiment:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = RobotNavEnv()
        self.category = 'RobotNav_Gym_LearningRate'

    def q_learning(self, evaluate=True):
        observation_space =self. env.observation_space.shape[0]
        action_space = self.env.action_space.n

        upper = self.env.observation_space.high
        lower = self.env.observation_space.low
        agent = Agent(action_space, upper=upper, lower=lower)

        params = {
            'method':'Baseline Q-Learning',
            'env_name':self.env_name,
            'gamma': agent.gamma,
            'learning_rate':agent.learning_rate,
            'eps': agent.exploration_rate,
            'eps_decay': agent.exploration_decay,
            'eps_min': agent.exploration_min,
            'batch_size': agent.batch_size,
            'bins': agent.model[0].getBins(),
            'epochs': agent.epochs,
            'category': self.category,
            'prioritized_experience_replay': False,
            'target_model_updates': 0
            }
        score_logger = ScoreLogger(params)

        start = time.process_time()
        for episode in range(1, 501):
            state = self.env.reset()
            state = np.reshape(state, [1, observation_space])
            step = 0
            rewards = 0
            while True:
                step += 1
                #env.render()
                action = agent.act(state)
                state_next, reward, done, _ = self.env.step(action)
                state_next = np.reshape(state_next, [1, observation_space])
                agent.remember(state, action, reward, state_next, done)
                state = state_next
                rewards += reward
                
                if done:
                    print('-------------------------------')
                    print("Episode: " + str(episode) + ", exploration: " + str(agent.exploration_rate) + ", score: " + str(step))
                    score_logger.add_score(round(rewards), episode)
                    if agent.model[0]._isFit:
                        print('tree sizes: ', [tree.get_depth() for tree in agent.model])
                    break
            
            for _ in tqdm(range(100)):
                agent.experience_replay()

            agent.exploration_rate *= agent.exploration_decay
            agent.exploration_rate = max(agent.exploration_min, agent.exploration_rate)

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
                rewards = 0
                while True:
                    step += 1
                    #env.render()
                    action = np.argmax([estimator.predict(state)[0] for estimator in agent.model])
                    print(f'Took action {action}')
                    for i, estimator in enumerate(agent.model):
                        print(f'SGT {i+1}:\nNode path: {estimator.tree.explain(state[0])}')
                    state_next, reward, done, _ = self.env.step(action)
                    state_next = np.reshape(state_next, [1, self.env.observation_space.shape[0]])
                    state = state_next
                    rewards += reward
                    if done:
                        print('-------------------------------')
                        print("Evaluation Episode: " + str(episode) + ", score: " + str(step))
                        score_logger.add_score(round(rewards), episode)
                        if agent.model[0]._isFit:
                            print('tree sizes: ', [tree.get_depth() for tree in agent.model])
                        break

if __name__ == "__main__":
    exp = Experiment('RobotNavigation')
    exp.q_learning()      