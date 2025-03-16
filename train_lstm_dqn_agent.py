import argparse
import numpy as np
import torch
from collections import deque
from simple_custom_taxi_env import SimpleTaxiEnv
from lstm_dqn import LSTMDQN

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

def train_lstm_dqn_agent(env, pretrained_model, num_episodes=10, batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9999):
    
    lstm_dqn = torch.load(pretrained_model) if pretrained_model is not None else LSTMDQN()
    epsilon = epsilon_start
    memory = deque(maxlen=10000)

    for episode in range(num_episodes):
        env = SimpleTaxiEnv(**env_config)
        obs, _ = env.reset()
        state = np.expand_dims(get_state(obs), axis=0)
        done = False
        total_reward = 0
        hidden = lstm_dqn.init_hidden(1)  # Initialize LSTM hidden state

        while not done:
            action = lstm_dqn.get_action(state, epsilon)

            obs, reward, done, _ = env.step(action)
            next_state = np.expand_dims(get_state(obs), axis=0)

            # Store experience
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Update model
            lstm_dqn.update(memory, batch_size, gamma)
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - Total reward: {total_reward:.2f} - Epsilon: {epsilon:.4f}")

    return lstm_dqn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid environment")
    parser.add_argument("--fuel_limit", type=int, default=5000, help="Maximum fuel available for the agent")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to the pretrained model for continued training")
    parser.add_argument("--save_path", type=str, default="lstm_dqn.pth", help="Path to save the training results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    env_config = {
        "grid_size": args.grid_size,
        "fuel_limit": args.fuel_limit
    }
    lstm_dqn = train_lstm_dqn_agent(env_config, args.pretrained_model)
    torch.save(lstm_dqn, "lstm_dqn.pth")
