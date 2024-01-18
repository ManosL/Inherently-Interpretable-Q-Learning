import argparse
import os
import sys
import json
import gym
import numpy as np
from sklearn.ensemble    import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
import torch             as T
import itertools

sys.path.append('../../')
sys.path.append('../../utils')

from NeuralNetworkTD3.td3_torch import Agent
from args_utils   import check_positive_int, check_dir_exists
from forecasting_metrics import mae, mse, mape

from util_functions import run_agent_for_episode_to_collect_experience
from util_functions import fit_gbr_from_agent_experience
from util_functions import run_gbr_for_episode



ENVIRONMENT_NAME               = 'InvertedPendulum-v4'
NUM_STEPS_TO_FIT_FROM          = 1e5
EPISODES_NUMBER                = 100
MAX_STEPS_PER_EPISODE          = 1000



def main(args):
    gbr_search_grid = {
        'n_estimators': [100, 250, 500, 750, 1000],
        'max_depth': [3, 5, 10, 20, 30, 45],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        'subsample': [0.25, 0.5, 0.75, 1.0],
        'min_samples_leaf': [1, 50, 100, 250, 500]
    }

    print('Running args and search grid are the following:')
    print(json.dumps(vars(args), sort_keys=True, indent=4))
    print(json.dumps(gbr_search_grid, sort_keys=True, indent=4))
    sys.stdout.flush()

    env = gym.make(args.env)

    agent = Agent(actor_lr=0.001, critic_lr=0.001, tau=0.05, input_dims=env.observation_space.shape, 
                                     env=env, n_actions=env.action_space.shape[0], warmup=0,
                                     chkpt_dir=args.model_dir, noise=0.1)

    agent.load_models()

    total_steps  = 0
    episodes_num = 0

    while total_steps < args.num_steps_to_fit_from:
        steps, score = run_agent_for_episode_to_collect_experience(agent, env, args.max_episode_duration)

        total_steps  += steps
        episodes_num += 1

        print(f'Collecting experience episode {episodes_num}(total steps {total_steps}) has reward {score}')
    

    best_model_mean   = None
    best_model_std    = None
    best_model_params = None

    for curr_gbr_kwargs in itertools.product(*gbr_search_grid.values()):
        gbr_kwargs = {}

        for gbr_arg_name, gbr_arg_curr_val in zip(gbr_search_grid.keys(), curr_gbr_kwargs):
            gbr_kwargs[gbr_arg_name] = gbr_arg_curr_val

        scores = []

        gbr_model = fit_gbr_from_agent_experience(agent, gbr_kwargs)


        for j in range(1, args.episodes + 1):
            steps, score = run_gbr_for_episode(agent, gbr_model, env, args.max_episode_duration)

            scores.append(score)

        curr_mean = np.array(scores).mean()
        curr_std = np.array(scores).std()
        
        if best_model_mean is None or (abs(best_model_mean - curr_mean) <= 300.0 and curr_std < best_model_std) \
            or (curr_mean > best_model_mean + 300):
            best_model_params = dict(gbr_kwargs)
            
            best_model_mean = curr_mean
            best_model_std  = curr_std 

        print_str  = f'GBR model with the following params\n{json.dumps(gbr_kwargs, sort_keys=True, indent=4)}\n'
        print_str += f'scored {curr_mean} units on an average of {args.episodes} episodes with std equal to {curr_std}.' 

        print(print_str)
        sys.stdout.flush()

    print_str = f'The best GBR model has the following params\n{json.dumps(best_model_params, sort_keys=True, indent=4)}\n'
    print_str += f'and yields mean score equal to {best_model_mean} with std equal to {best_model_std}.'
    print(print_str)
    print('-------------------------------------------------------------------------------------------------------')
    sys.stdout.flush()

    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainable TD3 trained using SGTs, XGBoost or GBR as mimic, i.e. replacing the target actor network with SGTs or XGBoost Trees.')

    parser.add_argument('--env', type=str, default=ENVIRONMENT_NAME,
                    help=f'OpenAI Gym environment to run our agent on.(default {ENVIRONMENT_NAME})')
    parser.add_argument('--episodes', type=check_positive_int, default=EPISODES_NUMBER,
                    help=f'Number of episodes that the gbr agent will be evaluated. (default {EPISODES_NUMBER})')
    parser.add_argument('--num_steps_to_fit_from', type=check_positive_int, default=NUM_STEPS_TO_FIT_FROM,
                    help=f'How many steps will be used to fit the GBR model. (default {NUM_STEPS_TO_FIT_FROM})')
    parser.add_argument('--max_episode_duration', type=check_positive_int, default=MAX_STEPS_PER_EPISODE,
                    help=f'Maximum steps per episode. (default {MAX_STEPS_PER_EPISODE})')
    parser.add_argument('--model_dir', type=check_dir_exists, default='./td3',
                    help=f'Directory that contains the weights of the TD3 models. Its your \
                     responsibility for those weights to correspond to the environment you provided. (default "./td3")')

    args = parser.parse_args()
    
    main(args)
