import argparse
import os
import sys
import json
import gym
import numpy as np
from sklearn.ensemble    import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
import torch             as T

sys.path.append('../../')
sys.path.append('../../utils')

from NeuralNetworkTD3.td3_torch import Agent
from args_utils   import check_positive_int, check_dir_exists, check_positive_float
from forecasting_metrics import mae, mse, mape

from util_functions import run_agent_for_episode_to_collect_experience
from util_functions import fit_gbr_from_agent_experience
from util_functions import run_gbr_for_episode



ENVIRONMENT_NAME               = 'InvertedPendulum-v4'
NUM_STEPS_TO_FIT_FROM          = 1e5
EPISODES_NUMBER                = 100
MAX_STEPS_PER_EPISODE          = 1000
GBR_ESTIMATORS                 = 100
GBR_MAX_DEPTH                  = 10000
GBR_LEARNING_RATE              = 0.1
GBR_SUBSAMPLE                  = 1.0
GBR_MIN_LEAF_SAMPLES           = 1



def main(args):
    scores = []

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
    
    print('Creating the GBR model')
    gbr_kwargs = {
        'n_estimators': args.gbr_estimators,
        'max_depth': args.gbr_max_depth,
        'learning_rate': args.gbr_learning_rate,
        'subsample': args.gbr_subsample,
        'min_samples_leaf': args.gbr_min_leaf_samples
    }

    gbr_model = fit_gbr_from_agent_experience(agent, gbr_kwargs)

    print('Starting evaluation of GBR model...')

    for j in range(1, args.episodes + 1):
        steps, score = run_gbr_for_episode(agent, gbr_model, env, args.max_episode_duration)

        scores.append(score)

        print('GBR model evaluation at episode', j, ': score %.1f' % score, f'steps: {steps}')
        sys.stdout.flush()

    avg_score = np.array(scores).mean()
    scores_std = np.array(scores).std()

    print(f'GBR model scored {avg_score} units on an average of {args.episodes} episodes.')
    print(f'95% confidence interval is +- {2 * scores_std}')

    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainable TD3 trained using SGTs, XGBoost or GBR as mimic, i.e. replacing the target actor network with SGTs or XGBoost Trees.')

    parser.add_argument('--env', type=str, default=ENVIRONMENT_NAME,
                    help=f'OpenAI Gym environment to run our agent on.(default {ENVIRONMENT_NAME})')
    parser.add_argument('--episodes', type=check_positive_int, default=EPISODES_NUMBER,
                    help=f'Number of episodes that the agent will be trained. (default {EPISODES_NUMBER})')
    parser.add_argument('--num_steps_to_fit_from', type=check_positive_int, default=NUM_STEPS_TO_FIT_FROM,
                    help=f'How many steps will be used to fit the GBR model. (default {NUM_STEPS_TO_FIT_FROM})')
    parser.add_argument('--max_episode_duration', type=check_positive_int, default=MAX_STEPS_PER_EPISODE,
                    help=f'Maximum steps per episode. (default {MAX_STEPS_PER_EPISODE})')
    parser.add_argument('--gbr_estimators', type=check_positive_int, default=GBR_ESTIMATORS,
                    help=f'How many estimators the GBR have. (default {GBR_ESTIMATORS})')
    parser.add_argument('--gbr_max_depth', type=check_positive_int, default=GBR_MAX_DEPTH,
                    help=f'Max depth of each GBR tree(estimator). (default {GBR_MAX_DEPTH})')
    parser.add_argument('--gbr_learning_rate', type=check_positive_float, default=GBR_LEARNING_RATE,
                    help=f'GBR\'s learning rate(default {GBR_LEARNING_RATE})')
    parser.add_argument('--gbr_subsample', type=check_positive_float, default=GBR_SUBSAMPLE,
                    help=f'GBR\'s percentage of subsample(default {GBR_SUBSAMPLE})')
    parser.add_argument('--gbr_min_leaf_samples', type=check_positive_int, default=GBR_MIN_LEAF_SAMPLES,
                    help=f'GBR\'s min number of samples that a leaf node should have(default {GBR_MIN_LEAF_SAMPLES})')

    parser.add_argument('--model_dir', type=check_dir_exists, default='./td3',
                    help=f'Directory that contains the weights of the TD3 models. Its your \
                     responsibility for those weights to correspond to the environment you provided. (default "./td3")')

    args = parser.parse_args()
    
    print('Running args are the following:')
    print(json.dumps(vars(args), sort_keys=True, indent=4))

    sys.stdout.flush()
    main(args)
