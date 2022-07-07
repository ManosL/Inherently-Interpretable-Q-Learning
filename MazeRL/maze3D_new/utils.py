import math
import numpy as np
from scipy.spatial import distance

from maze3D_new.config import left_down, right_down, left_up

goals = {"left_down": left_down, "left_up": left_up, "right_down": right_down}

ball_diameter = 43.615993


def checkTerminal(ball, goal):
    goal = goals[goal]
    if distance.euclidean([ball.x, ball.y], goal) < (ball_diameter/3):
        return True
    return False


def get_distance_from_goal(ball, goal):
    goal = goals[goal]
    return math.sqrt(math.pow(ball.x - goal[0], 2) + math.pow(ball.y - goal[1], 2))


def convert_actions(actions):
    # gets a list of 4 elements. it is called from getKeyboard()
    action = []
    if actions[0] == 1:
        action.append(1)
    elif actions[1] == 1:
        action.append(2)
    else:
        action.append(0)
    if actions[2] == 1:
        action.append(1)
    elif actions[3] == 1:
        action.append(2)
    else:
        action.append(0)
    return action
