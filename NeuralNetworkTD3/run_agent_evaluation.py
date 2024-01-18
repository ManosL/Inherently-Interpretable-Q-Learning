import argparse
import os
import sys
import json
import gym
import numpy as np
from td3_torch import Agent

sys.path.append('../utils')

from args_utils   import check_positive_int, check_dir_exists



ENVIRONMENT_NAME               = 'InvertedPendulum-v4'
EPISODES_NUMBER                = 100
MAX_STEPS_PER_EPISODE          = 1000



def evaluate_agent_in_episode(agent, env, max_steps):
    agent.eval() # Because we want to evaluate the agent

    observation, _ = env.reset()
    done  = False
    score = 0
    steps = 0

    while not done and steps < max_steps:
        action = agent.choose_action(observation)
        next_observation, reward, done, info, _ = env.step(action)

        score += reward
        steps += 1

        observation = next_observation

    return score



def main(args):
    scores = []

    env = gym.make(args.env)

    agent = Agent(actor_lr=0.001, critic_lr=0.001, tau=0.05, input_dims=env.observation_space.shape, 
                  env=env, n_actions=env.action_space.shape[0], chkpt_dir=args.model_dir)

    agent.load_models()

    for j in range(1, args.episodes + 1):
        score = evaluate_agent_in_episode(agent, env, args.max_episode_duration)

        scores.append(score)

        print('Pretrained model evaluation at episode', j, ': score %.1f' % score)
        sys.stdout.flush()

    avg_score = np.array(scores).mean()
    scores_std = np.array(scores).std()

    print(f'Pretrained agent scored {avg_score} units on an average of {args.episodes} episodes.')
    print(f'Standard Deviation is {scores_std}')

    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainable TD3 trained using SGTs, XGBoost or GBR as mimic, i.e. replacing the target actor network with SGTs or XGBoost Trees.')

    parser.add_argument('--env', type=str, default=ENVIRONMENT_NAME,
                    help=f'OpenAI Gym environment to run our agent on.(default {ENVIRONMENT_NAME})')
    parser.add_argument('--episodes', type=check_positive_int, default=EPISODES_NUMBER,
                    help=f'Number of episodes that the agent will be trained. (default {EPISODES_NUMBER})')
    parser.add_argument('--max_episode_duration', type=check_positive_int, default=MAX_STEPS_PER_EPISODE,
                    help=f'Maximum steps per episode. (default {MAX_STEPS_PER_EPISODE})')
    parser.add_argument('--model_dir', type=check_dir_exists, default='./td3',
                    help=f'Directory that contains the weights of the TD3 models. Its your \
                     responsibility for those weights to correspond to the environment you provided. (default "./td3")')

    args = parser.parse_args()
    
    print('Running args are the following:')
    print(json.dumps(vars(args), sort_keys=True, indent=4))

    sys.stdout.flush()
    main(args)
