# Remember to adjust your student ID in meta.xml
import numpy as np
import torch
from torch.distributions import Categorical
from utils import load_qtable

MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST, PICK_UP, DROP_OFF = 0, 1, 2, 3, 4, 5
ACTIONS_SPACE = [MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST, PICK_UP, DROP_OFF]
BACK_ACTIONS_SPACE = [MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST]

policy_table, stations_state, passenger_picked_up, action = None, None, None, None 
def policy_table_get_action(obs):
    from train_policy_table_agent import softmax, get_state
    global policy_table, stations_state, passenger_picked_up, action
    if policy_table is None:
        policy_table = load_qtable("results/policy_table1.pkl")
        stations_state = [0, 0, 0, 0]
        passenger_picked_up = [False]

    state = get_state(obs, stations_state, passenger_picked_up, action)
    if state in policy_table:
        action = np.random.choice(ACTIONS_SPACE, p=softmax(policy_table[state]))
    else:
        action = np.random.choice(ACTIONS_SPACE[:4])
    return action

lstmppo, hidden, stations_state, passenger_picked_up, action = None, None, None, None, None
def lstmppo_get_action(obs):
    from train_lstmppo_agent import get_state
    global lstmppo, hidden, stations_state, passenger_picked_up, action
    if lstmppo is None:
        lstmppo = torch.load("results/lstmppo.pth", weights_only=False)
        hidden = lstmppo.init_hidden()
        stations_state = [0, 0, 0, 0]
        passenger_picked_up = [False]

    state = get_state(obs, stations_state, passenger_picked_up, action)
    prob, hidden = lstmppo.pi(torch.from_numpy(np.array(state)).float(), hidden)
    prob = prob.view(-1)
    action = Categorical(prob).sample().item()
    return action

avoid_obstacles_qtable = None
def avoid_obstacles_qtable_get_action(obs):
    from train_avoid_obstacles_qtable_agent import get_state
    global avoid_obstacles_qtable
    if avoid_obstacles_qtable is None:
        avoid_obstacles_qtable = load_qtable("results/avoid_obstacles_qtable.pkl")
    state = get_state(obs)
    if state in avoid_obstacles_qtable:
        return np.argmax(avoid_obstacles_qtable[state])
    return np.random.choice(ACTIONS_SPACE[:4])

qtable, stations_state, passenger_picked_up, action = None, None, None, None 
def qtable_get_action(obs):
    from train_qtable_agent import get_state
    global qtable, stations_state, passenger_picked_up, action
    if qtable is None:
        qtable = load_qtable("results/qtable.pkl")
        stations_state = [0, 0, 0, 0]
        passenger_picked_up = [False]    
    state = get_state(obs, stations_state, passenger_picked_up, action)
    action = np.random.choice(ACTIONS_SPACE[:4]) if state not in qtable else np.argmax(qtable[state])
    return action

avoid_obstacles_qtable, qtable, stations_state, passenger_picked_up, action = None, None, None, None, None
def get_action(obs):
    from train_avoid_obstacles_qtable_agent import get_state as avoid_obstacles_get_state
    from train_qtable_agent import get_state

    global avoid_obstacles_qtable, qtable, stations_state, passenger_picked_up, action
    if avoid_obstacles_qtable is None or qtable is None:
        avoid_obstacles_qtable = load_qtable("results/avoid_obstacles_qtable.pkl")
        qtable = load_qtable("results/qtable.pkl")
        stations_state = [0, 0, 0, 0]
        passenger_picked_up = [False]

    avoid_obstacles_action_scores = avoid_obstacles_qtable[avoid_obstacles_get_state(obs)]
    action_scores = qtable[get_state(obs, stations_state, passenger_picked_up, action)]
    mix_scores = avoid_obstacles_action_scores + action_scores
    if action in BACK_ACTIONS_SPACE:
        mix_scores[BACK_ACTIONS_SPACE[action]] = -101
    action = np.argmax(mix_scores)
    action_scores[action] -= 1
    return action
