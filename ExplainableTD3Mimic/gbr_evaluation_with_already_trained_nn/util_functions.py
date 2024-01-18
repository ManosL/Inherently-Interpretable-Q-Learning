import sys
from sklearn.ensemble    import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
import torch             as T

sys.path.append('../')
sys.path.append('../utils')


from forecasting_metrics import mse


def run_agent_for_episode_to_collect_experience(agent, env, max_steps):
    #agent.eval() # Because we want to evaluate the agent

    observation, _ = env.reset()
    done  = False
    score = 0
    steps = 0

    while not done and steps < max_steps:
        action = agent.choose_action(observation) # Choose actions with some noise
        next_observation, reward, done, info, _ = env.step(action)
        
        # Verify that after load the replay memory is empty
        agent.remember(observation, action, reward, next_observation, done)

        score += reward
        steps += 1

        observation = next_observation

    return steps, score



def fit_gbr_from_agent_experience(agent, gbr_kwargs):
    gbr_model = MultiOutputRegressor(GradientBoostingRegressor(**gbr_kwargs))

    states, _, _, _, _ = agent.memory.sample_all()

    # Get NN's actions without any noise
    states          = T.tensor(states, dtype=T.float).to(agent.critic_1.device)
    actor_actions  = agent.actor(states)

    actor_actions  = actor_actions.cpu().detach().clone()

    states_numpy  = states.cpu().detach().numpy()
    actions_numpy = actor_actions.cpu().detach().numpy()

    # print(np.mean(states_numpy, axis=0), np.std(states_numpy, axis=0))
    # print(np.mean(actions_numpy, axis=0), np.std(actions_numpy, axis=0))
    gbr_model.fit(states_numpy, actions_numpy)

    print('FIT MSE', mse(actions_numpy, gbr_model.predict(states_numpy)))

    return gbr_model



def run_gbr_for_episode(agent, gbr_model, env, max_steps):
    n_actions = agent.n_actions

    observation, _ = env.reset()
    done  = False
    score = 0
    steps = 0

    while not done and steps < max_steps:
        action = gbr_model.predict([observation])
        action = action.reshape((-1, n_actions))[0]

        next_observation, reward, done, info, _ = env.step(action)

        score += reward
        steps += 1

        observation = next_observation

    return steps, score
