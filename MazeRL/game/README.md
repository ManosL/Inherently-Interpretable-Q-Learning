## Human Input
The human input is given to the system through the keyboard arrows and can be either discrete or continuous. 
* Discrete input means that every keystroke produces one action. For another action, the human needs to release the button and press another one (or the same). It needs 5 keystrokes to reach the maximum rotation towards each direction from the starting pose of the tray.
* Continuous input means that a new human action is given as long as a key is pressed. For a pressed key, a new action will be available every ~15ms.

## Experiments
The main functions for running the game exist in the streaming_gradient_q-learning.py
* `Single Agent experiments.` As a first interaction with the Maze-RL environment, the setting of training one SGTs based agent was selected. In that setting, a single agent is trained to solve the game by controlling the maze axes towards making the ball reach the hole target.
* `Two Agents collaborative experiments.` The next set of experiments took advantage of the Maze-RL environment design, as a collaborative setting for two players, each one of them controlling one of the two maze axes towards moving the ball at the hole target.
* `Human – AI collaborative experiments.` The third experimental setting is including human intervention in the collaboration with the agent. Specifically, an SGT agent is trained to control the x maze axis (Up, Down maze movement), while the human controls the y axis using the right and left arrow keys. Again, the goal is to move the ball to the hole target by collaboratively adjusting the axes angles.
* `Visual signal based explainable AI method.` SGT agents’ models trained during the two-agent collaboration training experiment were stored for later use. The individual evaluation of those two models, with a human user controlling the other axis, x or y depending on the case, demonstrated a smooth collaborative experience. The human along with the trained AI achieve effective collaboration, moving consistently the ball to the target within a very small number of game steps. The idea behind the creation of explanation signals is based on the fact that the two agents have been trained together to solve the task, hence the actions they have learnt to select at each game step are strongly correlated. This led to the hypothesis that the agent that is replaced by the human players in the Human – AI collaboration task, could point them towards the action that it could select on the exact same game step.


## Rewards
The following reward functions exist in the rewards.py:
* `reward_function_timeout_penalty`: For every non-terminal state the agent receives a reward of -1. In the goal state the agent receives a reward of 100. If the episode ends due to timeout, the agent gets -50.
* `reward_function_shafti`: For every non-terminal state the agent receives a reward of -1. In the goal state the agent receives a reward of 10.
* `reward_function_distance`: For every non-terminal state the agent receives a reward based on its distance from the goal. In the goal state the agent receives a reward of 100. If the episode ends due to timeout, the agent gets -50.
* `reward_function`: Template function in order to write a custom reward function.




