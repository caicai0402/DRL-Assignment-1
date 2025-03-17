import argparse
import torch
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from lstmppo import LSTMPPO
from simple_custom_taxi_env import SimpleTaxiEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid environment")
    parser.add_argument("--fuel_limit", type=int, default=5000, help="Maximum fuel available for the agent")
    parser.add_argument("--obstacles_percentage", type=float, default=0.1, help="Percentage of grid occupied by obstacles (0.0 to 1.0)")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to the pretrained model for continued training")
    parser.add_argument("--save_path", type=str, default="lstmppo.pth", help="Path to save the training results")
    return parser.parse_args()

def get_state(obs):
    return obs

def train_agent(env_config, pretrained_model=None, num_episodes=1000, horizon_steps=20):

    env = SimpleTaxiEnv(**env_config)
    model = torch.load(pretrained_model, weights_only=False) if pretrained_model is not None else LSTMPPO()
    
    rewards_per_episode = []
    for episode in tqdm(range(num_episodes)):
        # env_config["grid_size"] = np.random.randint(5, 30)
        # env = SimpleTaxiEnv(**env_config)
        
        obs, _ = env.reset()
        hidden_in = model.init_hidden()
        state = get_state(obs)
        
        done = False
        total_reward = 0
        
        while not done:
            for _ in range(horizon_steps):
                prob, hidden_out = model.pi(torch.from_numpy(np.array(state)).float(), hidden_in)
                prob = prob.view(-1)
                action = Categorical(prob).sample().item()

                obs, reward, done, _ = env.step(action)
                next_state = get_state(obs)
                model.put_data((state, action, reward / 100.0, next_state, prob[action].item(), hidden_in, hidden_out, done))
                
                if done:
                    break

                state = next_state
                hidden_in = hidden_out
                total_reward += reward
                
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
