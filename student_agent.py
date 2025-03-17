# Remember to adjust your student ID in meta.xml
import numpy as np
import torch
from torch.distributions import Categorical
from utils import load_q_table

MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST, PICK_UP, DROP_OFF = 0, 1, 2, 3, 4, 5
ACTIONS_SPACE = [MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST, PICK_UP, DROP_OFF]

q_table = None
def avoid_obstacles_get_action(obs):
    from train_avoid_obstacles_agent import get_state
    global q_table
    if q_table is None:
        q_table = load_q_table("./results/avoid_obstacles_q_table.pkl")

    state = get_state(obs)
    if state in q_table:
        return np.argmax(q_table[state])
    return np.random.choice(ACTIONS_SPACE[:4])

lstmppo, hidden = None, None
def lstmppo_get_action(obs):
    from train_lstmppo_agent import get_state
    global lstmppo, hidden
    if lstmppo is None:
        lstmppo = torch.load("results/lstmppo.pth", weights_only=False)
        hidden = lstmppo.init_hidden()

    state = get_state(obs)
    prob, hidden = lstmppo.pi(torch.from_numpy(np.array(state)).float(), hidden)
    prob = prob.view(-1)
    action = Categorical(prob).sample().item()
    return action

q_table, stations_state, passenger_picked_up, prev_action = None, None, None, None 
def get_action(obs):
    return avoid_obstacles_get_action(obs)
    # return lstmppo_get_action(obs)

    from train_agent import get_state
    global q_table, stations_state, passenger_picked_up, prev_action
    if q_table is None:
        q_table = load_q_table("q_table.pkl")
        stations_state = [0, 0, 0, 0]
        passenger_picked_up = [False]

    state = get_state(obs, stations_state, passenger_picked_up, prev_action)
    prev_action = action = np.random.choice(ACTIONS_SPACE[:4]) if state not in q_table else np.argmax(q_table[state])
    return action
