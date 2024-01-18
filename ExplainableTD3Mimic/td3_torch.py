import sys
sys.path.append('../utils')

import numpy               as np
import torch               as T
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
import os
import pickle

from pysgt.StochasticGradientTree import SGTRegressor
from xgboost                      import XGBRegressor
from sklearn.ensemble             import GradientBoostingRegressor
from sklearn.multioutput          import MultiOutputRegressor, RegressorChain

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions       import NotFittedError

from bounds        import ENV_BOUNDS
from replay_buffer import ReplayBuffer

from states        import WarmupState, TrainState
from states        import EvalState, ExperienceGainState

from forecasting_metrics import mae, mse, mape



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
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

        return



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



    def get_2nd_last_layer_output(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)

        return mu
    


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

        return



    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


  
class ExplainableTD3MimicAgent:
    # gamma = discount factor
    def __init__(self, actor_lr, critic_lr, input_dims, tau, env,
            gamma=0.99, update_actor_interval=2, update_target_model_interval=2, warmup=1000,
            n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300, 
            batch_size=100, target_model_batch_size=1024, target_model_batch_window=1024,
            exp_gain_steps = 1024, exp_gain_actors = 3, sgt_epochs=10, sgt_bins=8,
            sgt_batch_size=16, sgt_learning_rate=0.5, gbr_estimators=100, gbr_max_depth=10000, 
            gbr_learning_rate=0.1, gbr_subsample=1.0, gbr_min_leaf_samples=1, noise=0.1, 
            q_loss_reward_scale=1.0, target_model_sampling_strategy='recent', 
            target_model_type='sgt', fit_target_actor_without_tanh=False, 
            add_noise_to_target_actor_labels=False, return_fit_info=True, chkpt_dir='tmp/td3'):
        # Setting up the necessary variables
        self.gamma                            = gamma
        self.tau                              = tau
        self.env                              = env
        self.max_action                       = env.action_space.high
        self.min_action                       = env.action_space.low
        self.memory                           = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size                       = batch_size
        self.target_model_batch_size          = target_model_batch_size
        self.target_model_batch_window        = target_model_batch_window
        self.learn_step_cntr                  = 0
        self.time_step                        = 0
        self.warmup                           = warmup
        self.n_actions                        = n_actions
        self.update_actor_iter                = update_actor_interval
        self.update_target_model_iter         = update_target_model_interval
        self.target_model_sampling_strategy   = target_model_sampling_strategy
        self.target_model_type                = target_model_type
        self.fit_target_actor_without_tanh    = fit_target_actor_without_tanh
        self.add_noise_to_target_actor_labels = add_noise_to_target_actor_labels
        self.return_fit_info                  = return_fit_info

        #######################################################################
        # NEW ATTRIBUTES REGARDING EXPERIENCE_GAIN
        self.exp_gain_steps     = exp_gain_steps
        self.actors_to_sample   = exp_gain_actors
        self.exp_gain_memory    = ReplayBuffer(self.actors_to_sample * self.exp_gain_steps, input_dims, n_actions)
        self.curr_state         = WarmupState(self, 0, self.memory, self.warmup)
        self.before_eval_state  = None # I have this variable because the user can switch to eval state whenever he wants

        #######################################################################

        # Setting up the networks
        self.actor    = ActorNetwork(actor_lr, input_dims, layer1_size,
                            layer2_size, n_actions=n_actions, 
                            max_action=env.action_space.high, 
                            name='actor', chkpt_dir=chkpt_dir)
        
        self.critic_1 = CriticNetwork(critic_lr, input_dims, layer1_size,
                            layer2_size, n_actions=n_actions, 
                            name='critic_1', chkpt_dir=chkpt_dir)

        self.critic_2 = CriticNetwork(critic_lr, input_dims, layer1_size,
                            layer2_size, n_actions=n_actions, 
                            name='critic_2', chkpt_dir=chkpt_dir)
        
        # env_bounds = ENV_BOUNDS[env.unwrapped.spec.id.split('-')[0]]

        # Setting up target networks
        # No upper and lower bounds as all of the coordinates of observation space
        # do not have any bounds in MuJoCo envs. Later will see how will address it
        self.sgt_kwargs = {
            'epochs'        : sgt_epochs,
            'bins'          : sgt_bins,
            'batch_size'    : sgt_batch_size,
            'learning_rate' : sgt_learning_rate
        }

        self.gbr_kwargs = {
            'n_estimators': gbr_estimators,
            'max_depth': gbr_max_depth,
            'learning_rate': gbr_learning_rate,
            'subsample': gbr_subsample,
            'min_samples_leaf': gbr_min_leaf_samples
        }

        self.target_actor    = self.__initialize_target_network_model()
        
        # The following will be used just for insights
        self.target_actor_nn = ActorNetwork(actor_lr, input_dims, layer1_size,
                                        layer2_size, n_actions=n_actions, 
                                        max_action=env.action_space.high, 
                                        name='default', chkpt_dir=chkpt_dir) if self.return_fit_info else None

        ##############################################

        self.target_critic_1 = CriticNetwork(critic_lr, input_dims, layer1_size,
                                        layer2_size, n_actions=n_actions, 
                                        name='target_critic_1', chkpt_dir=chkpt_dir)

        self.target_critic_2 = CriticNetwork(critic_lr, input_dims, layer1_size,
                                        layer2_size, n_actions=n_actions, 
                                        name='target_critic_2', chkpt_dir=chkpt_dir)

        self.chkpt_dir = chkpt_dir
        self.noise = noise

        self.q_loss_reward_scale = q_loss_reward_scale
        self.q_loss_reward_scale_decay = 0.98
        self.q_loss_reward_scale_decay_freq = 10000

        self.update_network_parameters(tau=1)

        return



    """
    Define dynamically the SGT's bounds depending on the
    current batch that we will fit the SGT. It can be defined
    in 2 ways:

        -> if min_max == True then 
                upper = max(states, axis=0) and 
                lower = min(states, axis=0)

        -> if min_max == False then
                upper = mean(states, axis=0) + std_scale * std(states, axis=0)
                lower = mean(states, axis=0) - std_scale * std(states, axis=0)
    """
    def __find_sgt_bounds(self, states, min_max=False, std_scale=2):
        if min_max:
            return list(states.max(axis=0)), list(states.min(axis=0))
        
        means = states.mean(axis=0)
        stds  = states.std(axis=0)

        return list(means + (std_scale * stds)), list(means - (std_scale * stds))
    


    # states parameter is used for SGTs in order to find its new
    # bounds dynamically
    def __initialize_target_network_model(self, states=None):
        assert(self.target_model_type in ['sgt', 'gbr', 'xgboost'])

        if self.target_model_type == 'sgt':
            if states is not None:
                upper, lower = self.__find_sgt_bounds(states)

                self.sgt_kwargs['upper_bounds'] = upper
                self.sgt_kwargs['lower_bounds'] = lower

            return [SGTRegressor(**self.sgt_kwargs) for _ in range(self.n_actions)]
        elif self.target_model_type == 'gbr':
            print(self.gbr_kwargs)
            sys.stdout.flush()
            return MultiOutputRegressor(GradientBoostingRegressor(**self.gbr_kwargs))
        elif self.target_model_type == 'xgboost':
            return XGBRegressor(n_estimators=100, max_depth=None)

        return None



    def __target_network_model_fit(self, states, actions):
        assert(self.target_model_type in ['sgt', 'gbr', 'xgboost'])

        if self.target_model_type == 'sgt':
            [self.target_actor[i].fit(states, actions[:, i]) for i in range(self.n_actions)]
        elif self.target_model_type in ['xgboost', 'gbr']:
            print(f'STATES MEAN AND STD ARE\n{np.mean(states, axis=0)} AND {np.std(states, axis=0)}\n\n')
            print(f'ACTIONS MEAN AND STD ARE\n{np.mean(actions, axis=0)} AND {np.std(actions, axis=0)}\n\n')
            self.target_actor.fit(states, actions)

        return



    def __target_network_model_predict(self, states):
        assert(self.target_model_type in ['sgt', 'gbr', 'xgboost'])

        if self.target_model_type == 'sgt':
            states = np.array(states)
            preds = np.array([target_actor_tree.predict(states) for target_actor_tree in self.target_actor])
            preds = preds.reshape((-1, self.n_actions))
        elif self.target_model_type in ['xgboost', 'gbr']:
            preds = self.target_actor.predict(states)
            preds = preds.reshape((-1, self.n_actions))

        if self.fit_target_actor_without_tanh:
            preds = np.tanh(preds) * np.array(self.actor.max_action)

        return preds



    def __update_critics(self, state, action, reward, next_state, done):
        # Just for insights

        if self.return_fit_info:
            target_actions_nn = self.target_actor_nn(next_state)
            target_actions_nn = target_actions_nn.cpu().detach().numpy()

        ###################

        target_actions = self.__target_network_model_predict(next_state.cpu().detach().numpy())

        # Just for insights
        ex_model_target_actions = np.array(target_actions)

        # we consider as ground truth nn actor's actions
        nn_ex_model_actions_mae = mae(target_actions_nn, ex_model_target_actions) if self.return_fit_info else None
        ###################

        target_actions = T.tensor(target_actions, dtype=T.float).to(self.critic_1.device)
        
        # "Smoothing" target actions
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

        # Monitor these two losses(see pseudocode)
        q1_loss = F.mse_loss(target, q1) * self.q_loss_reward_scale
        q2_loss = F.mse_loss(target, q2) * self.q_loss_reward_scale

        # print(f'Q1 LOSS {q1_loss}')
        # print(f'Q2 LOSS {q2_loss}')
        # print('\n')

        critic_loss = q1_loss + q2_loss

        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        return nn_ex_model_actions_mae
    


    def __update_actor(self, state):
        # Update Actor Network
        self.actor.optimizer.zero_grad()

        actor_actions  = self.actor(state)
        actor_q1_loss  = self.critic_1(state, actor_actions)

        # In SGT will this work equally if we put as label as it is and all instances have the same one?
        actor_loss = -T.mean(actor_q1_loss)

        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return
    


    def train(self):
        if isinstance(self.curr_state, EvalState):
            if self.before_eval_state == None:
                self.curr_state = TrainState(self, self.memory)
            else:
                self.curr_state = self.before_eval_state

        return



    def eval(self):
        if not isinstance(self.curr_state, EvalState):
            self.before_eval_state = self.curr_state
            self.curr_state        = EvalState(self) 

        return
    


    def is_on_exp_gain(self):
        return isinstance(self.curr_state, ExperienceGainState)
    

    
    def choose_random_action(self, with_noise):
        mu = T.tensor(self.env.action_space.sample()).to(self.actor.device) #T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)

        if with_noise:
            mu_prime = mu + T.tensor(np.random.normal(scale=self.noise * self.max_action[0]), dtype=T.float).to(self.actor.device)
            mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        else:
            mu_prime = mu

        return mu_prime.cpu().detach().numpy()
    


    def choose_actor_action(self, observation, with_noise):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu    = self.actor.forward(state).to(self.actor.device)

        if with_noise:
            mu_prime = mu + T.tensor(np.random.normal(scale=self.noise * self.max_action[0]), dtype=T.float).to(self.actor.device)
            mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        else:
            mu_prime = mu

        return mu_prime.cpu().detach().numpy()



    # This is useful only on EVALUATION
    def choose_actor_action_from_target(self, observation):
        assert(isinstance(self.curr_state, EvalState))

        return self.__target_network_model_predict([observation])[0]



    def choose_action(self, observation, from_target_actor=False):
        return self.curr_state.choose_action(observation, from_target_actor)
    


    def remember(self, state, action, reward, next_state, done):
        self.curr_state.remember(state, action, reward, next_state, done)

        return
    

    def learn_step(self):
        return self.curr_state.learn_step()
    


    def change_state(self, new_state):
        self.curr_state = new_state
        return
    


    def change_to_train_state(self):
        self.change_state(TrainState(self, self.memory))
        return
    


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return (None, None, None)
        
        state, action, reward, next_state, done = \
                self.memory.sample_buffer(self.batch_size)
                
        reward     = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done       = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state      = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action     = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        # Fit the target actor adhoc if is not fitted before, just to output something in xgboost case
        if self.target_model_type != 'sgt':
            try:
                check_is_fitted(self.target_actor)
            except NotFittedError:
                self.__target_network_model_fit(state.cpu().detach().numpy(), action.cpu().detach().numpy())

        actions_mae = self.__update_critics(state, action, reward, next_state, done)

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.q_loss_reward_scale_decay_freq:
            if self.q_loss_reward_scale > 1.00:
                self.q_loss_reward_scale = max(1.00, self.q_loss_reward_scale * self.q_loss_reward_scale_decay)
    
        if self.learn_step_cntr % self.update_actor_iter == 0:
            self.__update_actor(state)

        fit_mse, fit_mape = None, None

        if self.learn_step_cntr % self.update_target_model_iter == 0:
            if self.target_model_sampling_strategy == 'exp_gain':
                self.change_state(ExperienceGainState(self, self.exp_gain_memory, self.exp_gain_steps, self.curr_state))
            else:
                fit_mse, fit_mape = self.update_target_actor()

        return (actions_mae, fit_mse, fit_mape)



    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params    = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        target_actor_params    = self.target_actor_nn.named_parameters() if self.return_fit_info else None
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor    = dict(actor_params) if self.return_fit_info else None
 
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)
        target_actor    = dict(target_actor_params) if self.return_fit_info else None

        for name in critic_1:
            critic_1[name] = tau * critic_1[name].clone() + \
                    (1 - tau) * target_critic_1[name].clone()
        
        for name in critic_2:
            critic_2[name] = tau * critic_2[name].clone() + \
                    (1 - tau) * target_critic_2[name].clone()

        if self.return_fit_info:
            for name in actor:
                actor[name] = tau * actor[name].clone() + \
                        (1 - tau) * target_actor[name].clone()
            
        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)

        if self.return_fit_info:
            self.target_actor_nn.load_state_dict(actor)

        return


    
    def update_target_actor(self):
        #if self.memory.mem_cntr < self.target_model_batch_size:
        #    return (None, None)
        
        if self.target_model_sampling_strategy == 'recent':
            states, _, _, _, _ = self.memory.sample_buffer_most_recent(self.target_model_batch_size)
        elif self.target_model_sampling_strategy == 'recent_window':
            states, _, _, _, _ = self.memory.sample_buffer_from_most_recent_window(self.target_model_batch_size, self.target_model_batch_window)
        elif self.target_model_sampling_strategy == 'exp_gain':
            assert(self.exp_gain_memory.is_full())
            states, _, _, _, _ = self.exp_gain_memory.sample_buffer(self.target_model_batch_size)
        else:
            assert(self.target_model_sampling_strategy == 'all')
            states, _, _, _, _ = self.memory.sample_buffer(self.target_model_batch_size)

        states          = T.tensor(states, dtype=T.float).to(self.critic_1.device)

        if self.fit_target_actor_without_tanh:
            actor_actions = self.actor.get_2nd_last_layer_output(states)
        else:
            actor_actions  = self.actor(states)

        actor_actions  = actor_actions.cpu().detach().clone()

        states_numpy  = states.cpu().detach().numpy()
        actions_numpy = actor_actions.cpu().detach().numpy()

        if self.add_noise_to_target_actor_labels:
            target_actions_numpy = self.__target_network_model_predict(states_numpy)

            actions_numpy = self.tau * actions_numpy + (1 - self.tau) * target_actions_numpy

        self.target_actor    = self.__initialize_target_network_model(states_numpy)

        self.__target_network_model_fit(states_numpy, actions_numpy)

        if self.return_fit_info:
            pred_actions = self.__target_network_model_predict(states_numpy)

            if self.fit_target_actor_without_tanh:
                return (mse(np.tanh(actor_actions.numpy()) * np.array(self.actor.max_action), np.tanh(pred_actions) * np.array(self.actor.max_action)), 
                        mape(np.tanh(actor_actions.numpy()) * np.array(self.actor.max_action), np.tanh(pred_actions) * np.array(self.actor.max_action)))
            
            return (mse(actor_actions.numpy(), pred_actions), mape(actor_actions.numpy(), pred_actions))

        return (None, None)



    def save_models(self):
        self.actor.save_checkpoint()
        # Because target_actor is an SGT just save it in a pickle file
        
        target_actor_file = open(os.path.join(self.chkpt_dir, 'target_actor.pkl'), 'wb')
        pickle.dump(self.target_actor, target_actor_file)
        target_actor_file.close()

        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

        return
    


    def load_models(self):
        self.actor.load_checkpoint()
        # Because target_actor is an SGT just load it from a pickle file
        
        target_actor_file = open(os.path.join(self.chkpt_dir, 'target_actor.pkl'), 'rb')
        self.target_actor = pickle.load(target_actor_file)

        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

        return
