import argparse
import torch
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from lstmppo import LSTMPPO
from simple_custom_taxi_env import SimpleTaxiEnv

MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST, PICK_UP, DROP_OFF = 0, 1, 2, 3, 4, 5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid environment")
    parser.add_argument("--fuel_limit", type=int, default=5000, help="Maximum fuel available for the agent")
    parser.add_argument("--obstacles_percentage", type=float, default=0.1, help="Percentage of grid occupied by obstacles (0.0 to 1.0)")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to the pretrained model for continued training")
    parser.add_argument("--save_path", type=str, default="lstmppo.pth", help="Path to save the training results")
    return parser.parse_args()

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

def train_agent(env_config, pretrained_model=None, num_episodes=1000, horizon_steps=20):

    env = SimpleTaxiEnv(**env_config)
    model = torch.load(pretrained_model, weights_only=False) if pretrained_model is not None else LSTMPPO()
    
    rewards_per_episode = []
    for episode in tqdm(range(num_episodes)):
        # env_config["grid_size"] = np.random.randint(5, 30)
        # env = SimpleTaxiEnv(**env_config)
        
        obs, _ = env.reset()
        hidden_in = model.init_hidden()
        stations_state = [0, 0, 0, 0]
        passenger_picked_up = [False]
        state = get_state(obs, stations_state, passenger_picked_up, None)
        
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            for _ in range(horizon_steps):
                prob, hidden_out = model.pi(torch.from_numpy(np.array(state)).float(), hidden_in)
                prob = prob.view(-1)
                action = Categorical(prob).sample().item()

                obs, reward, done, _ = env.step(action)
                next_state = get_state(obs, stations_state, passenger_picked_up, action)
                
                if action in [MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST]:
                    if state[action]:
                        reward -= 1000
                    elif (
                        ((state[4], state[5]) == (1, 0) and action == MOVE_SOUTH) or
                        ((state[4], state[5]) == (-1, 0) and action == MOVE_NORTH) or 
                        ((state[4], state[5]) == (0, 1) and action == MOVE_EAST) or
                        ((state[4], state[5]) == (0, -1) and action == MOVE_WEST)
                    ):
                        reward += 10 - step_count / 100
                        step_count = 0
                    elif abs(state[4]) + abs(state[5]) > abs(next_state[4]) + abs(next_state[5]):
                        reward += 10
                elif action == PICK_UP:
                    if state[6] == False and next_state[6] == True:
                        reward += 50 - step_count / 100
                        step_count = 0
                    else:
                        reward -= 1000
                elif action == DROP_OFF:
                    if done:
                        reward += 50 - step_count / 100
                        step_count = 0
                    else:
                        reward -= 1000
                
                model.put_data((state, action, reward / 100.0, next_state, prob[action].item(), hidden_in, hidden_out, done))
                state = next_state
                hidden_in = hidden_out
                total_reward += reward
                step_count += 1
                if done:
                    break
                
            model.train_net()

        rewards_per_episode.append(total_reward)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"🚀 Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}")

    return model

if __name__ == "__main__":
    args = parse_args()
    env_config = {
        "grid_size": args.grid_size,
        "fuel_limit": args.fuel_limit,
        "obstacles_percentage": args.obstacles_percentage
    }
    lstmppo = train_agent(env_config, args.pretrained_model)
    torch.save(lstmppo, args.save_path)
