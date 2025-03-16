# Remember to adjust your student ID in meta.xml
import numpy as np
from utils import load_q_table

MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST, PICK_UP, DROP_OFF = 0, 1, 2, 3, 4, 5
ACTIONS_SPACE = [MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST, PICK_UP, DROP_OFF]
q_table = load_q_table()

def avoid_obstacle_get_action(obs):
    def get_state(obs):
        obstacle_north, obstacle_south, obstacle_east, obstacle_west = obs[10], obs[11], obs[12], obs[13]
        return (obstacle_south, obstacle_north, obstacle_east, obstacle_west)
    state = get_state(obs)
    if state in q_table:
        return np.argmax(q_table[state])
    return np.random.choice(ACTIONS_SPACE[:4])

def todo_get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    def get_state(obs):
        taxi_rol, taxi_col = obs[0], obs[1]
        station1_row, station1_col = obs[2], obs[3]
        station2_row, station2_col = obs[4], obs[5]
        station3_row, station3_col = obs[6], obs[7]
        station4_row, station4_col = obs[8], obs[9]
        obstacle_north, obstacle_south, obstacle_east, obstacle_west = obs[10], obs[11], obs[12], obs[13]
        passenger_look, destination_look = obs[14], obs[15]
        return (
            station1_row - taxi_rol, station1_col - taxi_col,
            station2_row - taxi_rol, station2_col - taxi_col,
            station3_row - taxi_rol, station3_col - taxi_col,
            station4_row - taxi_rol, station4_col - taxi_col,
            obstacle_north, obstacle_south, obstacle_east, obstacle_west,
            passenger_look, destination_look
        )
    
    state = get_state(obs)
    
    # obstacle_north, obstacle_south, obstacle_east, obstacle_west = obs[10], obs[11], obs[12], obs[13]
    # if obstacle_north:
    #     return MOVE_NORTH
    # if obstacle_south:
    #     return MOVE_SOUTH
    # if obstacle_east:
    #     return MOVE_EAST
    # if obstacle_west:
    #     return MOVE_WEST
    
    if state in q_table:
        return np.argmax(q_table[state])
    
    return np.random.choice(ACTIONS_SPACE[:4])
   

def get_action(obs):
    return avoid_obstacle_get_action(obs)
