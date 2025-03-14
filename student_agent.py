# Remember to adjust your student ID in meta.xml
import numpy as np
from utils import load_q_table

q_table = load_q_table()
def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    MOVE_SOUTH = 0
    MOVE_NORTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    PICK_UP = 4
    DROP_OFF = 5
    actions = [MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST, PICK_UP, DROP_OFF]
    
    if obs in q_table:
        return np.argmax(q_table[obs])
    return np.random.choice(actions[:4])
   