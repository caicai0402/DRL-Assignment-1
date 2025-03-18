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

def get_state(obs):
    obstacle_north, obstacle_south, obstacle_east, obstacle_west = obs[10], obs[11], obs[12], obs[13]
    return (obstacle_south, obstacle_north, obstacle_east, obstacle_west)

def train_agent(env_config, pretrained_model=None, num_episodes=1000, alpha=1.0, gamma=0.0, 
                epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999):
    
    env = SimpleTaxiEnv(**env_config)
    qtable = {} if pretrained_model is None else load_qtable(pretrained_model)

    epsilon = epsilon_start
    rewards_per_episode = []
    
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        state = get_state(obs)
        done = False
        total_reward = 0
        while not done:
            if state not in qtable:
                qtable[state] = np.zeros(env.action_space_size)
                qtable[state][PICK_UP] = qtable[state][DROP_OFF] = -1000000000
               
            if np.random.rand() < epsilon:
                action = np.random.choice([MOVE_SOUTH, MOVE_NORTH, MOVE_EAST, MOVE_WEST])
            else:
                action = np.argmax(qtable[state])
            
            obs, reward, done, _ = env.step(action)
            next_state = get_state(obs)

            if action in [PICK_UP, DROP_OFF] or state[action]:
                reward = -1000000000
            else:
                reward = 0
            total_reward += reward
                      
            if next_state not in qtable:
                qtable[next_state] = np.ones(env.action_space_size)
                qtable[next_state][PICK_UP] = qtable[next_state][DROP_OFF] = -1000000000
            qtable[state][action] += alpha * (reward + gamma * np.max(qtable[next_state]) - qtable[state][action])
            
            state = next_state
            
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        rewards_per_episode.append(total_reward)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"🚀 Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    return qtable

if __name__ == "__main__":
    args = parse_args()
    env_config = {
        "grid_size": args.grid_size,
        "fuel_limit": args.fuel_limit,
        "obstacles_percentage": args.obstacles_percentage
    }
    qtable = train_agent(env_config, args.pretrained_model)
    store_qtable(qtable, args.save_path)
