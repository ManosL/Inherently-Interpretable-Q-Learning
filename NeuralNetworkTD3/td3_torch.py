import sys
sys.path.append('../utils')

import numpy               as np
import torch               as T
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
import os

from replay_buffer import ReplayBuffer



class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions,
            name, chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()

        self.learning_rate   = lr
        self.input_dims      = input_dims
        self.fc1_dims        = fc1_dims
        self.fc2_dims        = fc2_dims
        self.n_actions       = n_actions
        self.name            = name
        self.checkpoint_dir  = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        # Creating the network
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims + n_actions)
        self.fc2 = nn.Linear(self.fc1_dims + n_actions, self.fc2_dims)
        self.q1  = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device    = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

        return



    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim = 1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1



    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

        return



    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))



class ActorNetwork(nn.Module):
    # It is assumed that the domain of actions is of the form
    # [-max_action, +max_action]
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
            n_actions, max_action, name, chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()

        # max_action is used in order to get the correct range of actions
        #
        # n_actions is not the possible number of actions that the agent
        # can take, but how many "action" spaces we need to predict.
        self.learning_rate   = lr
        self.input_dims      = input_dims
        self.fc1_dims        = fc1_dims
        self.fc2_dims        = fc2_dims
        self.n_actions       = n_actions
        self.max_action      = max_action
        self.name            = name
        self.checkpoint_dir  = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        # Creating the network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu  = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device    = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

        return


    
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob)) * T.tensor(self.max_action).to(self.device)

        return mu



    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

        return



    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))



class Agent:
    # gamma = discount factor
    def __init__(self, actor_lr, critic_lr, input_dims, tau, env,
            gamma=0.99, update_actor_interval=2, warmup=1000,
            n_actions=2, max_size=1000000, layer1_size=400,
            layer2_size=300, batch_size=100, noise=0.1,
            chkpt_dir='tmp/td3'):
        # Setting up the necessary variables
        self.gamma             = gamma
        self.tau               = tau
        self.env               = env
        self.max_action        = env.action_space.high
        self.min_action        = env.action_space.low
        self.memory            = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size        = batch_size
        self.learn_step_cntr   = 0
        self.time_step         = 0
        self.warmup            = warmup
        self.n_actions         = n_actions
        self.update_actor_iter = update_actor_interval
        self.eval_mode         = False

        # Setting up the networks
        self.actor    = ActorNetwork(actor_lr, input_dims, layer1_size,
                            layer2_size, n_actions=n_actions, 
                            max_action=env.action_space.high, name='actor', 
                            chkpt_dir=chkpt_dir)
        
        self.critic_1 = CriticNetwork(critic_lr, input_dims, layer1_size,
                            layer2_size, n_actions=n_actions, name='critic_1', 
                            chkpt_dir=chkpt_dir)

        self.critic_2 = CriticNetwork(critic_lr, input_dims, layer1_size,
                            layer2_size, n_actions=n_actions, name='critic_2', 
                            chkpt_dir=chkpt_dir)
        
        # Setting up target networks
        self.target_actor    = ActorNetwork(actor_lr, input_dims, layer1_size,
                                        layer2_size, n_actions=n_actions, 
                                        max_action=env.action_space.high, 
                                        name='target_actor', chkpt_dir=chkpt_dir)
        
        self.target_critic_1 = CriticNetwork(critic_lr, input_dims, layer1_size,
                                        layer2_size, n_actions=n_actions, 
                                        name='target_critic_1', chkpt_dir=chkpt_dir)

        self.target_critic_2 = CriticNetwork(critic_lr, input_dims, layer1_size,
                                        layer2_size, n_actions=n_actions, 
                                        name='target_critic_2', chkpt_dir=chkpt_dir)

        self.noise = noise
        self.update_network_parameters(tau=1)

        return



    def train(self):
        self.eval_mode = False

        return



    def eval(self):
        self.eval_mode = True

        return


        
    def choose_action(self, observation):
        if self.time_step < self.warmup and not self.eval_mode:
            mu = T.tensor(self.env.action_space.sample()).to(self.actor.device) # T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)

            if not self.eval_mode:
                self.time_step += 1
        else:
            if self.time_step == self.warmup:
                self.time_step += 1
                print('WARMUP ENDED')

            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu    = self.actor.forward(state).to(self.actor.device)

        if not self.eval_mode and self.time_step >= self.warmup:
            mu_prime = mu + T.tensor(np.random.normal(scale=self.noise * self.max_action[0]), dtype=T.float).to(self.actor.device)
            mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        else:
            mu_prime = mu

        return mu_prime.cpu().detach().numpy()


    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

        return


    
    def learn(self):
        if self.eval_mode:
            print('Called learn method when having the agent at evaluation mode. Doing nothing....')
            return

        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, next_state, done = \
                self.memory.sample_buffer(self.batch_size)
        
        reward     = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done       = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state      = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action     = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor(next_state)

        # "Smoothing" target actions
        # torch.randn_like(action) * self.policy_noise
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=0.2 * self.max_action[0])), -0.5 * self.max_action[0], 0.5 * self.max_action[0])
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        target_q1 = self.target_critic_1(next_state, target_actions)
        target_q2 = self.target_critic_2(next_state, target_actions)

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        target_q1[done] = 0.0
        target_q2[done] = 0.0

        target_q1 = target_q1.view(-1)
        target_q2 = target_q2.view(-1)

        target_critic_value = T.min(target_q1, target_q2)

        target = reward + self.gamma * target_critic_value
        target = target.view(self.batch_size, 1)

        # Update Critic Networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        print(f'Q1 LOSS {q1_loss}')
        print(f'Q2 LOSS {q2_loss}')
        print('\n')
        critic_loss = q1_loss + q2_loss

        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        # Update Actor Network

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1(state, self.actor(state))

        # In SGT will this work equally if we put as label as it is and all instances have the same one?
        actor_loss = -T.mean(actor_q1_loss)

        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return



    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params    = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        target_actor_params    = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor    = dict(actor_params)

        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)
        target_actor    = dict(target_actor_params)

        for name in critic_1:
            critic_1[name] = tau * critic_1[name].clone() + \
                    (1 - tau) * target_critic_1[name].clone()
        
        for name in critic_2:
            critic_2[name] = tau * critic_2[name].clone() + \
                    (1 - tau) * target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau * actor[name].clone() + \
                    (1 - tau) * target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

        return


    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

        return
    


    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

        return
