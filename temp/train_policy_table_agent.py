import argparse
import numpy as np
from tqdm import tqdm
from simple_custom_taxi_env import SimpleTaxiEnv
from utils import load_qtable, store_qtable

MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST, PICK_UP, DROP_OFF = 0, 1, 2, 3, 4, 5
ACTIONS_SPACE = [MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST, PICK_UP, DROP_OFF]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid environment")
    parser.add_argument("--fuel_limit", type=int, default=5000, help="Maximum fuel available for the agent")
    parser.add_argument("--obstacles_percentage", type=float, default=0.1, help="Percentage of grid occupied by obstacles (0.0 to 1.0)")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to the pretrained model for continued training")
    parser.add_argument("--save_path", type=str, default="qtable.pkl", help="Path to save the training results")
    return parser.parse_args()

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def get_state(obs, stations_state, passenger_picked_up, prev_action):
    taxi_pos = (obs[0], obs[1])
    stations_offset = [
        (obs[2] - taxi_pos[0], obs[3] - taxi_pos[1]),
        (obs[4] - taxi_pos[0], obs[5] - taxi_pos[1]),
        (obs[6] - taxi_pos[0], obs[7] - taxi_pos[1]),
        (obs[8] - taxi_pos[0], obs[9] - taxi_pos[1])
    ]
    obstacle_south, obstacle_north, obstacle_east, obstacle_west = obs[11], obs[10], obs[12], obs[13]
    passenger_look, destination_look = obs[14], obs[15]
    stations_dis = [stations_state[idx] + abs(station_offset[0]) + abs(station_offset[1]) for idx, station_offset in enumerate(stations_offset)]

    target = np.argmin(stations_dis)
    if stations_dis[target] == 0:
        stations_dis[target] = stations_state[target] = 1 << 30    
        if not passenger_picked_up[0] and passenger_look:
            if prev_action == PICK_UP:
                passenger_picked_up[0] = True
            else:
                stations_dis[target] = stations_state[target] = 0
        elif destination_look:
            stations_state[target] = 777777
        
    if passenger_picked_up[0] and 777777 in stations_state:
        target = stations_state.index(777777)
    else:
        target = np.argmin(stations_dis)
        
    if passenger_picked_up[0]:
        return (obstacle_south, obstacle_north, obstacle_east, obstacle_west,
                stations_offset[target][0], stations_offset[target][1],
                passenger_picked_up[0], destination_look)
    else:
        return (obstacle_south, obstacle_north, obstacle_east, obstacle_west,
                stations_offset[target][0], stations_offset[target][1],
                passenger_picked_up[0], passenger_look)

def train_agent(env_config, pretrained_model=None, num_episodes=10000, alpha=0.99, gamma=0.01):
    
    env = SimpleTaxiEnv(**env_config)
    policy_table = {} if pretrained_model is None else load_qtable(pretrained_model)

    rewards_per_episode = []
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()

        stations_state = [0, 0, 0, 0]
        passenger_picked_up = [False]
        state = get_state(obs, stations_state, passenger_picked_up, None)
        done = False
        trajectory = []
        total_reward = 0
        step_count = 0
        while not done:    
            if state not in policy_table:
                policy_table[state] = -1000000000 * np.array([state[0], state[1], state[2], state[3], 1, 1], dtype=np.float64)
                
            if (state[4], state[5]) == (0, 0) and state[6] and state[7]:
                policy_table[state][DROP_OFF] = 0
                action = DROP_OFF
            elif (state[4], state[5]) == (0, 0) and not state[6] and state[7]:
                policy_table[state][PICK_UP] = 0
                action = PICK_UP
            else:
                action = np.random.choice(ACTIONS_SPACE, p=softmax(policy_table[state]))
            
            obs, reward, done, _ = env.step(action)
            next_state = get_state(obs, stations_state, passenger_picked_up, action)

            ## reward shaping
            if action in [MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST]:
                if state[action]:
                    reward -= 1000
                elif (
                    ((state[4], state[5]) == (1, 0) and action == MOVE_SOUTH) or
                    ((state[4], state[5]) == (-1, 0) and action == MOVE_NORTH) or 
                    ((state[4], state[5]) == (0, 1) and action == MOVE_EAST) or
                    ((state[4], state[5]) == (0, -1) and action == MOVE_WEST)
                ):
                    reward += 100 - step_count / 10
                    step_count = 0
                elif abs(state[4]) + abs(state[5]) > abs(next_state[4]) + abs(next_state[5]):
                    reward += 10
            elif action == PICK_UP:
                if state[6] == False and next_state[6] == True:
                    reward += 500 - step_count / 10
                    step_count = 0
                else:
                    reward -= 1000
            elif action == DROP_OFF:
                if done:
                    reward += 500 - step_count / 10
                    step_count = 0
                else:
                    reward -= 1000

            total_reward += reward
            if trajectory == [] or (state, action, reward) != trajectory[-1]:
                trajectory.append((state, action, reward))

            step_count += 1

        G = 0  # Return (discounted sum of rewards)
        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = reward + gamma * G  # Discounted reward

            # Update policy table using policy gradient
            action_probs = softmax(policy_table[state])
            policy_gradient = np.zeros_like(policy_table[state])
            policy_gradient[action] = 1
            policy_table[state] += alpha * G * (policy_gradient - action_probs)
            
        rewards_per_episode.append(total_reward)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"🚀 Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}")

    return policy_table

if __name__ == "__main__":
    args = parse_args()
    env_config = {
        "grid_size": args.grid_size,
        "fuel_limit": args.fuel_limit,
        "obstacles_percentage": args.obstacles_percentage
    }
    policy_table = train_agent(env_config, args.pretrained_model)
    store_qtable(policy_table, args.save_path)
