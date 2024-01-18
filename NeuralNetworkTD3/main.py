import argparse
import os
import sys
import gym
import pybullet_envs
import numpy as np
from td3_torch import Agent
import math
import torch
import random
import pickle



sys.path.append('../utils')

from graph_utils   import plot_learning_and_eval_curve, plot_eval_curve
from args_utils    import check_positive_float, check_positive_int, check_dir_exists



ENVIRONMENT_NAME      = 'InvertedPendulum-v4'
ACTOR_LEARNING_RATE   = 0.001
CRITIC_LEARNING_RATE  = 0.001
TAU                   = 0.005
BATCH_SIZE            = 100
DISCOUNT_FACTOR       = 0.99
ACTOR_UPDATE_INTERVAL = 2
WARMUP_STEPS_NUMBER   = 1000
MAX_MEMORY_SIZE       = 1000000
LAYER_1_SIZE          = 400
LAYER_2_SIZE          = 300
EXPERIMENTS_NUMBER    = 5
MAX_TIMESTEPS         = 1e6
MAX_STEPS_PER_EPISODE = 1000
NOISE                 = 0.1
EVAL_FREQUENCY        = 5e3
EVAL_EPISODES         = 10



def run_agent_for_episode(agent, env, max_steps, seed, eval_mode=False):
    if eval_mode:
        agent.eval()
    else:
        agent.train()

    observation, _ = env.reset(seed=seed)
    done  = False
    score = 0
    steps = 0
    
    while not done and steps < max_steps:
        action = agent.choose_action(observation)
        next_observation, reward, done, info, _ = env.step(action)
        
        if not eval_mode:
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()

        score += reward
        steps += 1

        observation = next_observation

    return score, steps



def main(args):
    plot_filename = args.output_curves_path
    filename      = os.path.join('plots', plot_filename)

    train_score_history = []
    eval_score_history  = []
    eval_timesteps      = [min(args.eval_freq * i, args.max_timesteps) for i in range(math.ceil(args.max_timesteps / args.eval_freq) + 1)]
    i = 0
    curr_seed = 1
    global_best_score = -np.inf

    #agent.load_models()
    while i < args.experiments:
        env = gym.make(args.env) 
        best_score = env.reward_range[0]
    
        warmup_eps = 0
        warmup_avg_score = 0

        # Set seeds
        curr_seed += 1
        env.action_space.seed(curr_seed)
        random.seed(curr_seed)
        torch.manual_seed(curr_seed)
        np.random.seed(curr_seed)

        agent = Agent(actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                input_dims=env.observation_space.shape, tau=args.tau, env=env, gamma=args.discount_factor,
                update_actor_interval=args.actor_update_freq, warmup=args.warmup, 
                batch_size=args.batch_size, layer1_size=args.layer_1_size, layer2_size=args.layer_2_size,
                n_actions=env.action_space.shape[0], max_size=args.mem_size, noise=args.noise, 
                chkpt_dir=args.checkpoint_dir)

        train_score_history.append([])
        eval_score_history.append([])

        curr_episode = 0
        total_steps = 0
        next_eval_step_no = 0
        stuck_steps      = 0
        fallen_in_minima = False

        while total_steps <= args.max_timesteps:
            score, steps = run_agent_for_episode(agent, env, args.max_episode_duration, curr_seed, eval_mode=False)
            
            train_score_history[-1].append(score)
            avg_score = np.mean(train_score_history[-1][-50:])

            if total_steps <= args.warmup:
                warmup_eps += 1
                warmup_avg_score += score
            
            total_steps += steps

            # If we just got out of warmup
            if total_steps > args.warmup and total_steps - steps <= args.warmup:
                warmup_avg_score = warmup_avg_score / warmup_eps
                print('WARMUP ENDED', warmup_avg_score)

            if len(train_score_history[-1]) > 1 and train_score_history[-1][-1] <= warmup_avg_score:
                stuck_steps += 1

                if stuck_steps == 500:
                    stuck_steps = 0
                    fallen_in_minima = True
                    break
            else:
                stuck_steps = 0

            print('Total Steps', total_steps, 'Training Experiment', i , 'Episode', curr_episode, 'score %.1f' % score, 'average score %.1f' % avg_score)
            sys.stdout.flush()
            
            curr_episode += 1

            if total_steps >= args.max_timesteps or total_steps >= next_eval_step_no:
                if len(eval_score_history[-1]) == len(eval_timesteps):
                    continue
                
                next_eval_step_no += args.eval_freq
                eval_avg_score = 0

                for k in range(args.eval_episodes):
                    score, _ = run_agent_for_episode(agent, env, args.max_episode_duration, curr_seed + 100, eval_mode=True)
                    
                    eval_avg_score += score
                
                eval_avg_score /= args.eval_episodes
                eval_score_history[-1] += [eval_avg_score] #* (args.eval_freq - (j % args.eval_freq))

                if eval_avg_score > best_score:
                    best_score = eval_avg_score
                    agent.save_models()

                if eval_avg_score > global_best_score:
                    global_best_score = eval_avg_score
                    
                print('Evaluation returned average score equal to %.1f' % eval_avg_score)

        if fallen_in_minima:
            print('Fallen to a local minima in this experiment, thus we will repeat it')
            print('\n\n\n\n')

            train_score_history.pop()
            eval_score_history.pop()
        else:
            eval_curve_file = open(os.path.join(args.eval_curves_dir, f'curve_{args.experiments - i}.pkl'), 'wb')
            pickle.dump(eval_score_history[-1], eval_curve_file)
            eval_curve_file.close()
            
            i = i + 1

    plot_eval_curve(eval_score_history, eval_timesteps, filename, 
                    f'Evaluation scores on average of {args.eval_episodes} episodes per {args.eval_freq} steps')

    #x = [i + 1 for i in range(args.episodes)]
    #plot_learning_and_eval_curve(x, train_score_history, eval_score_history, filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TD3 Implementation with Deep Neural Networks.')

    parser.add_argument('--env', type=str, default=ENVIRONMENT_NAME,
                    help=f'OpenAI Gym environment to run our agent on.(default {ENVIRONMENT_NAME})')
    parser.add_argument('--actor_lr', type=check_positive_float, default=ACTOR_LEARNING_RATE,
                    help=f'Actor Learning Rate.(default {ACTOR_LEARNING_RATE})')
    parser.add_argument('--critic_lr', type=check_positive_float, default=CRITIC_LEARNING_RATE,
                    help=f'Critics\' Learning Rate.(default {CRITIC_LEARNING_RATE})')
    parser.add_argument('--tau', type=check_positive_float, default=TAU,
                    help=f'Tau number i.e. how much of a copy we will have to target neural networks.(default {TAU})')
    parser.add_argument('--batch_size', type=check_positive_int, default=BATCH_SIZE,
                    help=f'Batch size of neural networks.(dafault {BATCH_SIZE})')
    parser.add_argument('--discount_factor', type=check_positive_float, default=DISCOUNT_FACTOR,
                    help=f'Agent\'s discount factor.(default {DISCOUNT_FACTOR})')
    parser.add_argument('--actor_update_freq', type=check_positive_int, default=ACTOR_UPDATE_INTERVAL,
                    help=f'The update frequency(in terms of learning steps) of actor network. (default {ACTOR_UPDATE_INTERVAL})')
    parser.add_argument('--warmup', type=check_positive_int, default=WARMUP_STEPS_NUMBER,
                    help=f'The number of agent\'s warmup steps where it will just choose random actions. (default {WARMUP_STEPS_NUMBER})')
    parser.add_argument('--mem_size', type=check_positive_int, default=MAX_MEMORY_SIZE,
                    help=f'Size of Replay Buffer. (default {MAX_MEMORY_SIZE})')
    parser.add_argument('--layer_1_size', type=check_positive_int, default=LAYER_1_SIZE,
                    help=f'Number of neurons in first layer of the actor and critic networks. (default {LAYER_1_SIZE})')
    parser.add_argument('--layer_2_size', type=check_positive_int, default=LAYER_2_SIZE,
                    help=f'Number of neurons in second layer of the actor and critic networks. (default {LAYER_2_SIZE})')
    parser.add_argument('--experiments', type=check_positive_int, default=EXPERIMENTS_NUMBER,
                    help=f'How many experiments will be conducted. (default {EXPERIMENTS_NUMBER})')
    parser.add_argument('--max_timesteps', type=check_positive_int, default=MAX_TIMESTEPS,
                    help=f'Number of timesteps that the agent will be trained. (default {MAX_TIMESTEPS})')
    parser.add_argument('--max_episode_duration', type=check_positive_int, default=MAX_STEPS_PER_EPISODE,
                    help=f'Maximum steps per episode. (default {MAX_STEPS_PER_EPISODE})')
    parser.add_argument('--noise', type=check_positive_float, default=NOISE,
                    help=f'Noise added to agent\'s action when in training. (default {NOISE})')
    parser.add_argument('--eval_freq', type=check_positive_int, default=EVAL_FREQUENCY,
                    help=f'Evaluation frequency per training timesteps. (default {EVAL_FREQUENCY})')
    parser.add_argument('--eval_episodes', type=check_positive_int, default=EVAL_EPISODES,
                    help=f'Number of episodes that the agent will be evaluated per evalution. (default {EVAL_EPISODES})')
    parser.add_argument('--output_curves_path', type=str, default='./output.png',
                    help=f'Path to save the plot with training and evaluation learning curves. (default output.png)')
    parser.add_argument('--checkpoint_dir', type=check_dir_exists, default='./td3',
                    help=f'Path to save the weights of agent\'s best models that were found at evaluation. (default ./td3)')
    parser.add_argument('--eval_curves_dir', type=check_dir_exists, default='./eval_curves',
                    help=f'Path where we save inside the evaluation curves of each experiment. (default ./eval_curves)')
    
    args = parser.parse_args()
    
    print('Running args are the following:')
    print(args)
    main(args)
