import argparse
import os
import sys
import json
import gymnasium as gym
import pybullet_envs
import pickle
import numpy as np
from td3_torch import ExplainableTD3MimicAgent
import torch
import random
import math

sys.path.append('../utils')

from graph_utils  import plot_learning_and_eval_curve, plot_fit_curve, plot_eval_curve
from args_utils   import check_positive_float, check_positive_int, check_dir_exists
from args_utils   import check_target_model_sampling_strategy, check_target_model_type


######################## TRAIN AND NN RELATED ARGS #####################################

ENVIRONMENT_NAME               = 'InvertedPendulum-v4'
ACTOR_LEARNING_RATE            = 0.001
CRITIC_LEARNING_RATE           = 0.001
TAU                            = 0.005
BATCH_SIZE                     = 100  # Try greater batch size or try a greater new random batch
DISCOUNT_FACTOR                = 0.99
ACTOR_UPDATE_INTERVAL          = 2
WARMUP_STEPS_NUMBER            = 1000
MAX_MEMORY_SIZE                = 1000000
LAYER_1_SIZE                   = 400
LAYER_2_SIZE                   = 300
########################################################################################

######################## TARGET MODEL RELATED ARGS #####################################

TARGET_MODEL_BATCH_SIZE        = 1024  
TARGET_MODEL_BATCH_WINDOW      = 2048
TARGET_MODEL_UPDATE_INTERVAL   = 10 #50
TARGET_MODEL_SAMPLING_STRATEGY = 'all'
TARGET_MODEL_TYPE              = 'sgt'

########################################################################################

# Train the explainable target model after some time that the agent performs well with
# a NN as a target actor network.(FOR LATER)

# Also implement Prioritized Experience Replay(Priority 1) in order to use it also
# to extract the batch for the target actor. THIS IS USEFUL BECAUSE IF YOU SEE THE
# GRAPHS, THE AGENT SEEMS TO "UNLEARN" WHICH MIGHT B PROBABLY DUE TO THE "RANDOM"
# SAMPLING

# Also, see how good of a fit each target model type does.

############################# SGT RELATED ARGS #####################################

SGT_EPOCHS        = 10
SGT_BINS          = 8
SGT_BATCH_SIZE    = 16
SGT_LEARNING_RATE = 0.5

####################################################################################

############################# GBR RELATED ARGS #####################################

GBR_ESTIMATORS = 100
GBR_MAX_DEPTH = 10000
GBR_LEARNING_RATE = 0.1
GBR_SUBSAMPLE = 1.0
GBR_MIN_LEAF_SAMPLES = 1

####################################################################################

######################## EXP_GAIN RELATED ARGS #####################################

EXPERIENCE_GAIN_STEPS  = 1024
EXPERIENCE_GAIN_ACTORS = 3

####################################################################################

######################## EXPERIMENT RESULTS RELATED ARGS ###########################

EXPERIMENTS_NUMBER             = 5
MAX_TIMESTEPS                  = 1e6
MAX_STEPS_PER_EPISODE          = 1000
NOISE                          = 0.1
Q_LOSS_REWARD_SCALE            = 1.0 # more like Q_LOSS_SCALE
EVAL_FREQUENCY                 = 5e3
EVAL_EPISODES                  = 10

####################################################################################



def get_agent_kwargs(args, env):
    agent_kwargs = {
                        "actor_lr":                         args.actor_lr, 
                        "critic_lr":                        args.critic_lr,
                        "input_dims":                       env.observation_space.shape, 
                        "tau":                              args.tau, 
                        "env":                              env, 
                        "gamma":                            args.discount_factor,
                        "update_actor_interval":            args.actor_update_freq, 
                        "update_target_model_interval":     args.target_model_update_freq, 
                        "warmup":                           args.warmup, 
                        "batch_size":                       args.batch_size, 
                        "target_model_batch_size":          args.target_model_batch_size, 
                        "target_model_batch_window":        args.target_model_batch_window,
                        "layer1_size":                      args.layer_1_size, 
                        "layer2_size":                      args.layer_2_size, 
                        "n_actions":                        env.action_space.shape[0], 
                        "max_size":                         args.mem_size, 
                        "noise":                            args.noise, 
                        "q_loss_reward_scale":              args.q_loss_reward_scale,
                        "exp_gain_steps":                   args.experience_gain_steps, 
                        "exp_gain_actors":                  args.experience_gain_actors, 
                        "sgt_epochs":                       args.sgt_epochs, 
                        "sgt_bins":                         args.sgt_bins,
                        "sgt_batch_size":                   args.sgt_batch_size, 
                        "sgt_learning_rate":                args.sgt_learning_rate,
                        "gbr_estimators":                   args.gbr_estimators, 
                        "gbr_max_depth":                    args.gbr_max_depth,
                        "gbr_learning_rate":                args.gbr_learning_rate,
                        "gbr_subsample":                    args.gbr_subsample,
                        "gbr_min_leaf_samples":             args.gbr_min_leaf_samples,
                        "target_model_sampling_strategy":   args.target_model_sampling_strategy,
                        "target_model_type":                args.target_model_type, 
                        "add_noise_to_target_actor_labels": args.add_noise_to_target_actor_labels,
                        "fit_target_actor_without_tanh":    args.fit_target_actor_without_tanh, 
                        "chkpt_dir":                        args.checkpoint_dir
                    }

    return agent_kwargs



def run_agent_for_episode(agent, env, max_steps, seed, eval_mode=False):
    if eval_mode:
        agent.eval()
    else:
        agent.train()

    actions_maes          = []
    fit_mses              = []
    fit_mapes             = []

    observation, _ = env.reset() # seed=seed)
    done  = False
    score = 0
    steps = 0
    non_exp_gain_steps = 0

    while not done and steps < max_steps:
        step_on_exp_gain = agent.is_on_exp_gain()

        action = agent.choose_action(observation)
        next_observation, reward, done, info, _ = env.step(action)

        agent.remember(observation, action, reward, next_observation, done)
        actions_mae, fit_mse, fit_mape = agent.learn_step()

        score += reward
        steps += 1

        if not step_on_exp_gain:
            non_exp_gain_steps += 1

        if actions_mae != None:
            actions_maes.append(actions_mae)

        if fit_mse != None:
            fit_mses.append(fit_mse)
            fit_mapes.append(fit_mape)

        observation = next_observation

    return score, non_exp_gain_steps, non_exp_gain_steps == 0, actions_maes, fit_mses, fit_mapes



def main(args):
    if (args.target_model_sampling_strategy == 'recent_window') and (args.target_model_batch_window < args.target_model_batch_size):
        print('Target Model\'s Batch Window should be greater or equal than Target Model\'s batch size.')
        return 1

    filename                     = os.path.join('plots', args.output_curves_path)
    mse_filename                 = os.path.join('mse',   args.output_curves_path)
    mape_filename                = os.path.join('mape',  args.output_curves_path)
    train_nn_emodel_mae_filename = os.path.join('mae',  args.output_curves_path)

    train_score_history   = []
    eval_score_history    = []
    train_fit_mses        = []
    train_fit_mapes       = []
    train_nn_emodel_maes  = []
    eval_timesteps        = [min(args.eval_freq * i, args.max_timesteps) for i in range(math.ceil(args.max_timesteps / args.eval_freq) + 1)]
    i = 0
    curr_seed = -1
    global_best_score = -np.inf

    #agent.load_models()
    while i < args.experiments:
        env = gym.make(args.env)#'InvertedPendulumBulletEnv-v0', apply_api_compatibility=True)
        best_score = -np.inf

        warmup_eps = 0
        warmup_avg_score = 0

        # Set seeds
        curr_seed += 1
        # env.action_space.seed(curr_seed)
        # random.seed(curr_seed)
        # torch.manual_seed(curr_seed)
        # np.random.seed(curr_seed)

        agent = ExplainableTD3MimicAgent(**get_agent_kwargs(args, env))

        train_score_history.append([])
        eval_score_history.append([])
        train_fit_mses.append([])
        train_fit_mapes.append([])
        train_nn_emodel_maes.append([])
        
        curr_episode = 0
        total_steps = 0
        next_eval_step_no = 0
        stuck_steps      = 0
        fallen_in_minima = False

        while total_steps <= args.max_timesteps:
            score, steps, only_exp_gain_ep, actions_maes, fit_mses, fit_mapes = run_agent_for_episode(agent, env, args.max_episode_duration, curr_seed, eval_mode=False)

            if fit_mses != []:
                train_fit_mses[-1]  += fit_mses
                train_fit_mapes[-1] += fit_mapes

            if actions_maes != []:
                train_nn_emodel_maes[-1] += actions_maes

            if only_exp_gain_ep:
                # print('only exp gain ep')
                # sys.stdout.flush()
                continue

            train_score_history[-1].append(score)
            avg_score = np.mean(train_score_history[-1][-50:])

            if total_steps <= args.warmup:
                warmup_eps += 1
                warmup_avg_score += score
            
            total_steps += steps

            # If we just got out of warmup
            if total_steps > args.warmup and total_steps - steps <= args.warmup:
                warmup_avg_score = warmup_avg_score / warmup_eps
                print('WARMUP ENDED', 2.5 * warmup_avg_score)

            if total_steps > args.warmup and len(train_score_history[-1]) > 1 and train_score_history[-1][-1] <= 2.5 * warmup_avg_score:
                stuck_steps += 1

                if stuck_steps == 800:
                    stuck_steps = 0
                    fallen_in_minima = True
                    break
            else:
                if stuck_steps > 0:
                    print(train_score_history[-1][-1], train_score_history[-1][-2], train_score_history[-1][-1] - train_score_history[-1][-2], 'UNSTUCK')

                stuck_steps = 0

            print('Total Steps', total_steps, 'Training Experiment', i , 'Episode', curr_episode, 'score %.1f' % score, 'average score %.1f' % avg_score)
            curr_episode += 1
            sys.stdout.flush()

            if total_steps >= args.max_timesteps or total_steps >= next_eval_step_no:
                if len(eval_score_history[-1]) == len(eval_timesteps):
                    continue
                
                next_eval_step_no += args.eval_freq
                eval_avg_score = 0

                for k in range(args.eval_episodes):
                    score, _, _, _, _, _ = run_agent_for_episode(agent, env, args.max_episode_duration, curr_seed + 100, eval_mode=True)
                    
                    eval_avg_score += score
                
                eval_avg_score /= args.eval_episodes
                eval_score_history[-1].append(eval_avg_score)

                if eval_avg_score > best_score:
                    best_score = eval_avg_score
                    agent.save_models()

                if eval_avg_score > global_best_score:
                    global_best_score = eval_avg_score

                print('Evaluation returned average score equal to %.1f' % eval_avg_score)
                sys.stdout.flush()
        
        if fallen_in_minima:
            print('Fallen to a local minima in this experiment, thus we will repeat it')
            print('\n\n\n\n')
            sys.stdout.flush()

            train_score_history.pop()
            eval_score_history.pop()

            train_fit_mses.pop()
            train_fit_mapes.pop()
            train_nn_emodel_maes.pop()
        else:
            eval_curve_file = open(os.path.join(args.eval_curves_dir, f'curve_{args.experiments - i}.pkl'), 'wb')
            pickle.dump(eval_score_history[-1], eval_curve_file)
            eval_curve_file.close()

            mse_curve_file = open(os.path.join(args.eval_curves_dir, f'mse_curve_{args.experiments - i}.pkl'), 'wb')
            pickle.dump(train_fit_mses[-1], mse_curve_file)
            mse_curve_file.close()

            mae_curve_file = open(os.path.join(args.eval_curves_dir, f'mae_curve_{args.experiments - i}.pkl'), 'wb')
            pickle.dump(train_nn_emodel_maes[-1], mae_curve_file)
            mae_curve_file.close()            

            i = i + 1

    plot_eval_curve(eval_score_history, eval_timesteps, filename, 
                    f'Evaluation scores on average of {args.eval_episodes} episodes per {args.eval_freq} steps')

    if len(train_fit_mapes):
        plot_fit_curve(train_fit_mses,  mse_filename,  'Average MSE per episode along with their standard deviation')
        plot_fit_curve(train_fit_mapes, mape_filename, 'Average MAPE per episode along with their standard deviation')
        plot_fit_curve(train_nn_emodel_maes, train_nn_emodel_mae_filename, 'Average MAE per episode between NN and Expl. model actor \ndecisions along with their \nstandard deviation')

    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainable TD3 trained using SGTs OR XGBoost as mimic, i.e. replacing the target actor network with SGTs or XGBoost Trees.')

    parser.add_argument('--env', type=str, default=ENVIRONMENT_NAME,
                    help=f'OpenAI Gym environment to run our agent on.(default {ENVIRONMENT_NAME})')
    parser.add_argument('--actor_lr', type=check_positive_float, default=ACTOR_LEARNING_RATE,
                    help=f'Neural Network Actor Learning Rate.(default {ACTOR_LEARNING_RATE})')
    parser.add_argument('--critic_lr', type=check_positive_float, default=CRITIC_LEARNING_RATE,
                    help=f'Neural Network Critics\' Learning Rate.(default {CRITIC_LEARNING_RATE})')
    parser.add_argument('--tau', type=check_positive_float, default=TAU,
                    help=f'Tau number i.e. how much of a copy we will have to target neural networks.(default {TAU})')
    parser.add_argument('--batch_size', type=check_positive_int, default=BATCH_SIZE,
                    help=f'Batch size of neural networks.(dafault {BATCH_SIZE})')
    parser.add_argument('--target_model_batch_size', type=check_positive_int, default=TARGET_MODEL_BATCH_SIZE,
                    help=f'Batch size of target actor, who is implemented using SGTs or XGBoost Trees.(dafault {TARGET_MODEL_BATCH_SIZE})')
    parser.add_argument('--target_model_batch_window', type=check_positive_int, default=TARGET_MODEL_BATCH_WINDOW,
                    help=f'Batch Most Recent Window to sample from for Explainable target actor. It is used only if --strategy is recent_window(default {TARGET_MODEL_BATCH_WINDOW})')
    parser.add_argument('--discount_factor', type=check_positive_float, default=DISCOUNT_FACTOR,
                    help=f'Agent\'s discount factor.(default {DISCOUNT_FACTOR})')
    parser.add_argument('--actor_update_freq', type=check_positive_int, default=ACTOR_UPDATE_INTERVAL,
                    help=f'The update frequency(in terms of learning steps) of actor network. (default {ACTOR_UPDATE_INTERVAL})')
    parser.add_argument('--target_model_update_freq', type=check_positive_int, default=TARGET_MODEL_UPDATE_INTERVAL,
                    help=f'The update frequency(in terms of learning steps) of target actor. (default {TARGET_MODEL_UPDATE_INTERVAL})')
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
    parser.add_argument('--q_loss_reward_scale', type=check_positive_float, default=Q_LOSS_REWARD_SCALE,
                    help=f'Multiplier to Q losses. (default {Q_LOSS_REWARD_SCALE})')
    parser.add_argument('--experience_gain_steps', type=check_positive_int, default=EXPERIENCE_GAIN_STEPS,
                    help=f'ADD DESCRIPTION (default {EXPERIENCE_GAIN_STEPS})')
    parser.add_argument('--experience_gain_actors', type=check_positive_int, default=EXPERIENCE_GAIN_ACTORS,
                    help=f'ADD DESCRIPTION (default {EXPERIENCE_GAIN_ACTORS})')
    
    #############################################################################################

    parser.add_argument('--sgt_epochs', type=check_positive_int, default=SGT_EPOCHS,
                    help=f'SGT\' train epochs(applicable if chosen SGT as target model) (default {SGT_EPOCHS})')
    parser.add_argument('--sgt_bins', type=check_positive_int, default=SGT_BINS,
                    help=f'Bins that each SGT\'s node will have(applicable if chosen SGT as target model)(default {SGT_BINS})')
    parser.add_argument('--sgt_batch_size', type=check_positive_int, default=SGT_BATCH_SIZE,
                    help=f'ADD DESCRIPTION (applicable if chosen SGT as target model)(default {SGT_BATCH_SIZE})')
    parser.add_argument('--sgt_learning_rate', type=check_positive_float, default=SGT_LEARNING_RATE,
                    help=f'SGT\'s learning rate(applicable if chosen SGT as target model)(default {SGT_LEARNING_RATE})')

    parser.add_argument('--gbr_estimators', type=check_positive_int, default=GBR_ESTIMATORS,
                    help=f'GBR\'s number of estimators(applicable if chosen GBR as target model) (default {GBR_ESTIMATORS})')
    parser.add_argument('--gbr_max_depth', type=check_positive_int, default=GBR_MAX_DEPTH,
                    help=f'Bins that each SGT\'s node will have(applicable if chosen SGT as target model)(default {GBR_MAX_DEPTH})')
    parser.add_argument('--gbr_learning_rate', type=check_positive_float, default=GBR_LEARNING_RATE,
                    help=f'GBR\'s learning rate(applicable if chosen GBR as target model) (default {GBR_LEARNING_RATE})')
    parser.add_argument('--gbr_subsample', type=check_positive_float, default=GBR_SUBSAMPLE,
                    help=f'GBR\'s percentage of subsample(applicable if chosen GBR as target model)(default {GBR_SUBSAMPLE})')
    parser.add_argument('--gbr_min_leaf_samples', type=check_positive_int, default=GBR_MIN_LEAF_SAMPLES,
                    help=f'GBR\'s min number of samples that a leaf node should have(applicable if chosen GBR as target model) (default {GBR_MIN_LEAF_SAMPLES})')

    
    ##################################################################################################
    parser.add_argument('--eval_freq', type=check_positive_int, default=EVAL_FREQUENCY,
                    help=f'Evaluation frequency per training timesteps. (default {EVAL_FREQUENCY})')
    parser.add_argument('--eval_episodes', type=check_positive_int, default=EVAL_EPISODES,
                    help=f'Number of episodes that the agent will be evaluated per evalution. (default {EVAL_EPISODES})')
    parser.add_argument('--target_model_sampling_strategy', type=check_target_model_sampling_strategy, default=TARGET_MODEL_SAMPLING_STRATEGY,
                    help=f'Explainable target model\'s sampling strategy in order to update target actor(default {TARGET_MODEL_SAMPLING_STRATEGY}). Can be \'recent\', \'recent_window\' and \'all\'.\
                        \'recent\' takes the most recent --target_model_batch_size samples from the replay buffer.\
                        \'recent_window\' takes a sample of --target_model_batch_size from --target_model_batch_window most recent samples from the replay buffer.\
                        \'exp_gain\' takes a sample of --target_model_batch_size random samples from the replay buffer but the replay buffer is a separate one.\
                        \'all\' just takes a random sample from the whole replay buffer.')
    parser.add_argument('--target_model_type', type=check_target_model_type, default=TARGET_MODEL_TYPE,
                    help=f'Explainable target model\'s type(default {TARGET_MODEL_TYPE}). Can be equal to \'sgt\', \'gbr\' or \'xgboost\', non-case-sensitive.')
    parser.add_argument('--output_curves_path', type=str, default='./output.png',
                    help=f'Path to save the plot with training and evaluation learning curves. (default output.png)')
    parser.add_argument('--checkpoint_dir', type=check_dir_exists, default='./td3',
                    help=f'Path to save the weights of agent\'s best models that were found at evaluation. (default ./td3)')
    parser.add_argument('--eval_curves_dir', type=check_dir_exists, default='./eval_curves',
                    help=f'Path where we save inside the evaluation curves of each experiment. (default ./eval_curves)')
    
    #############################################################################################
    
    parser.add_argument('--add_noise_to_target_actor_labels', default=False, action='store_true',
                        help="ADD DESCRIPTION (default False)")
    parser.add_argument('--fit_target_actor_without_tanh', default=False, action='store_true',
                        help="ADD DESCRIPTION (default False)")
    
    #############################################################################################

    args = parser.parse_args()
    
    print('Running args are the following:')
    print(json.dumps(vars(args), sort_keys=True, indent=4))

    sys.stdout.flush()
    main(args)
