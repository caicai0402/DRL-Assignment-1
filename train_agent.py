import argparse
import numpy as np
from simple_custom_taxi_env import SimpleTaxiEnv
from utils import load_q_table, store_q_table

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid environment")
    parser.add_argument("--fuel_limit", type=int, default=5000, help="Maximum fuel available for the agent")
    parser.add_argument("--obstacles_percentage", type=float, default=0.1, help="Percentage of grid occupied by obstacles (0.0 to 1.0)")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to the pretrained model for continued training")
    parser.add_argument("--save_path", type=str, default="q_table.pkl", help="Path to save the training results")
    return parser.parse_args()

def get_state(obs):
    obstacle_north, obstacle_south, obstacle_east, obstacle_west = obs[10], obs[11], obs[12], obs[13]
    return (obstacle_south, obstacle_north, obstacle_east, obstacle_west)

def train_agent(env_config, pretrained_model=None, num_episodes=2000, alpha=0.99, gamma=0.01, 
                epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9995):
    
    env = SimpleTaxiEnv(**env_config)
    q_table = {} if pretrained_model is None else load_q_table(pretrained_model)

    epsilon = epsilon_start
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = get_state(obs)
        
        done = False
        total_reward = 0

        while not done:    
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space_size)

            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space_size)
            else:
                action = np.argmax(q_table[state])
            
            obs, reward, done, _ = env.step(action)

            if action >= 4 or state[action]:
                reward = -1000
            total_reward += reward
            
            next_state = get_state(obs)            
            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space_size)
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        rewards_per_episode.append(total_reward)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"🚀 Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    return q_table

if __name__ == "__main__":
    args = parse_args()
    env_config = {
        "grid_size": args.grid_size,
        "fuel_limit": args.fuel_limit,
        "obstacles_percentage": args.obstacles_percentage
    }
    q_table = train_agent(env_config, args.pretrained_model)
    store_q_table(q_table, args.save_path)
