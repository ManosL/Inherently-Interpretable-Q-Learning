import sys, os
sys.path.append(r'./')
sys.path.append(r'./MazeRL')
import numpy as np
import time, sys
from tqdm.auto import tqdm
from utils.scores import ScoreLogger
from maze3D_new.Maze3DEnv import *
from maze3D_new.assets import *
from rl_models.SGTAgent import Agent
from plot_utils.plot_utils import get_config


class Experiment:
    def __init__(self, env_name):
        self.env_name = env_name
        self.config = get_config(sys.argv[1])
        
        self.goal = self.config["game"]["goal"]
        self.discrete = self.config['game']['discrete_input']
        if self.config['game']['agent_only']:
            self.env = Maze3DGym(config_file=sys.argv[1])
            self.category = 'Single_Agent'
        elif self.config['game']['second_agent']:
            self.env = Maze3DCollaborative(config_file=sys.argv[1])
            self.category = 'Agents_Collaborative'
        else:
            self.env = Maze3D(config_file=sys.argv[1])
            self.category = 'Human_Agent_Collaborative'
        self.human_actions = [0, 0]

    def q_learning(self):

        action_space = self.env.action_space.actions_number
            
        upper = np.array([175, 175, 3, 3, 0.5, 0.5, 0.1, 0.1])
        lower = -upper
        
        if self.config['game']['second_agent']:
            title = ' Agent Collaborative'

            agents = [Agent(action_space, upper=upper, lower=lower),
                Agent(action_space, upper=upper, lower=lower)]

            if self.config['game']['load_checkpoint']:
                path = 'results/'+self.config['game']['checkpoint_name']+'/'
                agents[1].load_model(path+'right_left.pkl')
                agents[0].load_model(path+'up_down.pkl')
                agents[0].exploration_rate = 0.01
                agents[1].exploration_rate = 0.01

            params = {
                'method':'Baseline Q-Learning'+title,
                'env_name':self.env_name,
                'gamma': agents[0].gamma,
                'learning_rate':agents[0].learning_rate,
                'eps': agents[0].exploration_rate,
                'eps_decay': agents[0].exploration_decay,
                'eps_min': agents[0].exploration_min,
                'batch_size': agents[0].batch_size,
                'bins': agents[0].model[0].getBins(),
                'epochs': agents[0].epochs,
                'category' : self.category,
                'prioritized_experience_replay': False,
                'target_model_updates': 0
                }
            score_logger = ScoreLogger(params)
            start = time.time()

            if self.config['game']['test_model']:
                self.agent_evaluate(agents, score_logger)
            else:
                self.collaborative_training_loop(agents, score_logger)    

            pg.quit()

            agents[0].save_model(score_logger.folder_path+'up_down.pkl')
            agents[1].save_model(score_logger.folder_path+'right_left.pkl')    

            print('Time taken: {} minutes'.format((time.time() - start)/60))
            print('Total nodes: ',[[estimator.get_total_nodes() for estimator in agent.model] for agent in agents])
        
        else:
            title = ''
            agent = Agent(action_space, upper=upper, lower=lower)
            
            if self.config['game']['load_checkpoint']:
                path = 'results/'+self.config['game']['checkpoint_name']+'/'
                try:
                    agent.load_model(path+'agent.pkl')
                except FileNotFoundError:
                    if self.config['game']['predict_user_action']:
                        self.guide_agent = Agent(action_space, upper=upper, lower=lower)
                        self.guide_agent.load_model(path+'right_left.pkl')
                    agent.load_model(path+'up_down.pkl')
                agent.exploration_rate = 0.01    
            
            params = {
                'method':'Baseline Q-Learning'+title,
                'env_name':self.env_name,
                'gamma': agent.gamma,
                'learning_rate':agent.learning_rate,
                'eps': agent.exploration_rate,
                'eps_decay': agent.exploration_decay,
                'eps_min': agent.exploration_min,
                'batch_size': agent.batch_size,
                'bins': agent.model[0].getBins(),
                'epochs': agent.epochs,
                'category' : self.category,
                'prioritized_experience_replay': False,
                'target_model_updates': 0
                }
            score_logger = ScoreLogger(params)

            start = time.time()             

            if self.config['game']['test_model']:
                self.agent_evaluate(agent, score_logger)
            else:
                self.training_loop(agent, score_logger)    

            pg.quit()

            agent.save_model(score_logger.folder_path+'agent.pkl')

            print('Time taken: {} minutes'.format((time.time() - start)/60))
            print('Total nodes: ',[estimator.get_total_nodes() for estimator in agent.model])

    
    def training_loop(self, agent, score_logger):
        observation_space = self.env.observation_shape[0]
        max_timesteps =  int(self.config['Experiment']['max_games_mode']['max_duration'])
        cycles = 1000
        
        for episode in range(1, self.config['Experiment']['max_games_mode']['max_episodes']):
            state = np.array(self.env.reset()[0])
            state = np.reshape(state, [1, observation_space])
            
            timedout = False
            step = 0
            rewards = 0

            for timestep in range(max_timesteps):
                #print(f'Step {timestep} of {max_timesteps}, {time.time() - start}')
                step += 1
                action = agent.act(state)

                state_next, reward, done, _, _, _ = self.env.step(action, timedout, self.goal, 0.2)
                state_next = np.reshape(state_next, [1, observation_space])
                agent.remember(state, action, reward, state_next, done)
                state = state_next
                rewards += reward

                if self.config['game']['agent_only']:
                    score_logger.add_action(action)
                else:
                    human_actions = self.env.get_human_action()
                    score_logger.add_action([action, human_actions[1]])
                
                for event in pg.event.get():
                    if event.type == pg.KEYDOWN:
                        if event.key == 115:
                            agent.save_model(score_logger.folder_path+'agent.pkl')

                score_logger.add_observation(state[0])

                if episode % 10 == 0 and (done or (timestep == max_timesteps-1)):
                    for _ in tqdm(range(cycles)):
                        agent.experience_replay()
                    if self.config['Experiment']['scheduling'] == 'descending':
                        cycles = round(cycles * 0.5)      
                
                if done or (timestep == max_timesteps-1):
                    print('-------------------------------')
                    print("Episode: " + str(episode) + ", exploration: " + str(agent.exploration_rate) + ", score: " + str(step))
                    print("Learning rate: ", agent.model[0].lr)
                    score_logger.add_score(round(rewards), episode)
                    if agent.model[0]._isFit:
                        print('tree sizes: ', [tree.get_depth() for tree in agent.model])
                    if not done:
                        timedout = True
                    break

            agent.exploration_rate *= agent.exploration_decay
            agent.exploration_rate = max(agent.exploration_min, agent.exploration_rate)
        

    def collaborative_training_loop(self, agents, score_logger):
        observation_space = self.env.observation_shape[0]
        max_timesteps =  int(self.config['Experiment']['max_games_mode']['max_duration'])
        cycles = 1000
        
        for episode in range(1, self.config['Experiment']['max_games_mode']['max_episodes']):
            state = np.array(self.env.reset()[0])
            state = np.reshape(state, [1, observation_space])
            
            timedout = False
            step = 0
            rewards = 0

            for timestep in range(max_timesteps):
                step += 1
                actions = [agent.act(state) for agent in agents]

                state_next, reward, done, _, _, _ = self.env.step(actions, timedout, self.goal, 0.2)
                state_next = np.reshape(state_next, [1, observation_space])
                [agent.remember(state, actions[i], reward, state_next, done) for i, agent in enumerate(agents)]
                state = state_next
                rewards += reward
                score_logger.add_action(actions)
                score_logger.add_observation(state[0])

                # if timestep % 100 == 0:
                if episode % 10 == 0 and (done or (timestep == max_timesteps-1)):
                    for _ in tqdm(range(cycles)):
                        [agent.experience_replay() for agent in agents]
                    if self.config['Experiment']['scheduling'] == 'descending':
                        cycles = round(cycles * 0.5)  
                
                if done or (timestep == max_timesteps-1):
                    print('-------------------------------')
                    print("Episode: " + str(episode) + ", exploration: " + str(agents[0].exploration_rate) + ", score: " + str(step))
                    print("Learning rate: ", agents[0].model[0].lr)
                    score_logger.add_score(round(rewards), episode)
                    if agents[0].model[0]._isFit:
                        print('tree sizes: ', [[tree.get_depth() for tree in agent.model] for agent in agents])
                    if not done:
                        timedout = True
                    break

            agents[0].exploration_rate *= agents[0].exploration_decay
            agents[0].exploration_rate = max(agents[0].exploration_min, agents[0].exploration_rate)

            agents[1].exploration_rate *= agents[1].exploration_decay
            agents[1].exploration_rate = max(agents[1].exploration_min, agents[1].exploration_rate)

    
    def agent_evaluate(self, agent, score_logger):
        max_timesteps = int(self.config['Experiment']['max_games_mode']['max_duration'])
            
        for episode in range(1, self.config['Experiment']['max_games_mode']['max_evaluation_episodes']):
                state = np.array(self.env.reset()[0])
                state = np.reshape(state, [1, self.env.observation_shape[0]])
                if self.config['game']['second_agent']:
                        action = [a.act(state) for a in agent]
                else:
                    action = np.argmax([model.predict(state)[0] for model in agent.model])
                step = 0
                rewards = 0
                timedout = False

                for timestep in range(max_timesteps):
                    if self.config['game']['predict_user_action']:
                        if os.path.exists(f"results/{self.config['game']['checkpoint_name']}/right_left.pkl"):
                            estimated_human_action = np.argmax([guide_model.predict(state)[0] for guide_model in self.guide_agent.model]) 
                            self.env.board.update()
                            glClearDepth(1000.0)
                            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                            self.env.board.draw(mode=estimated_human_action+4)
                            pg.display.flip()
                            time.sleep(0.07)
                    step += 1
                    #env.render()
                    if self.config['game']['second_agent']:
                        action = [a.act(state) for a in agent]
                    else:
                        action = np.argmax([estimator.predict(state)[0] for estimator in agent.model])
                    
                    state_next, reward, done, _, _, _ = self.env.step(action, timedout, self.goal, 0.2)
                    state_next = np.reshape(state_next, [1, self.env.observation_shape[0]])
                    state = state_next
                    rewards += reward

                    if self.config['game']['agent_only']:
                        score_logger.add_action(action)
                    elif self.config['game']['second_agent']:
                        score_logger.add_action(action)
                    else:
                        human_actions = self.env.get_human_action()
                        score_logger.add_action([action, human_actions[1]])
                    score_logger.add_observation(state[0])

                    if done or (timestep == max_timesteps-1):
                        print('-------------------------------')
                        print("Evaluation Episode: " + str(episode) + ", score: " + str(step))
                        score_logger.add_score(rewards, episode)
                        if self.config['game']['second_agent']:
                            if agent[0].model[0]._isFit:
                                print('tree sizes: ', [[tree.get_depth() for tree in a.model] for a in agent])
                        else:
                            if agent.model[0]._isFit:
                                print('tree sizes: ', [tree.get_depth() for tree in agent.model])
                        if not done:
                            timedout = True
                        break

if __name__ == "__main__":
    exp = Experiment('Maze3D')
    exp.q_learning()
    exit(0)