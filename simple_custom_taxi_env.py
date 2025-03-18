import argparse
import importlib.util
import time
from IPython.display import clear_output
import random

# This environment allows you to verify whether your program runs correctly during testing, 
# as it follows the same observation format from `env.reset()` and `env.step()`. 
# However, keep in mind that this is just a simplified environment. 
# The full specifications for the real testing environment can be found in the provided spec.
# 
# You are free to modify this file to better match the real environment and train your own agent. 
# Good luck!

class SimpleTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=5000, obstacles_percentage=0.1):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.obstacles_percentage = obstacles_percentage
        self.action_space_size = 6
    
    def generate_stations(self, num_stations):
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        random.shuffle(all_positions)
        stations = []
        for x, y in all_positions:
            if any(abs(x - sx) + abs(y - sy) == 1 for sx, sy in stations):  
                continue
            stations.append((x, y))
            if len(stations) == num_stations:
                break
        return stations
        
    def generate_obstacles(self, obstacle_ratio):
        num_obstacles = int(self.grid_size * self.grid_size * obstacle_ratio)
        obstacles = set()
        while len(obstacles) < num_obstacles:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in self.stations:
                obstacles.add(pos)
        return obstacles

    def reset(self):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False

        self.stations = self.generate_stations(4)
        self.obstacles = self.generate_obstacles(self.obstacles_percentage)

        available_positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.stations and (x, y) not in self.obstacles
        ]

        self.taxi_pos = random.choice(available_positions)
        self.passenger_loc, self.destination = random.sample(self.stations, 2)
        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos  
                else:
                    reward -= 10  
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward -0.1, True, {}
                    else:
                        reward -= 10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -= 10
                    
        reward -= 0.1  

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward -10, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle
    
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    
    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        grid[self.stations[0][0]][self.stations[0][1]]='R'
        grid[self.stations[1][0]][self.stations[1][1]]='G'
        grid[self.stations[2][0]][self.stations[2][1]]='Y'
        grid[self.stations[3][0]][self.stations[3][1]]='B'
        
        # Place passenger
        if self.passenger_loc is not None:
            py, px = self.passenger_loc
            if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                grid[py][px] = 'P'
        
        # Place destination
        if self.destination is not None:
            dy, dx = self.destination
            if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
                grid[dy][dx] = 'D'
        
        # Place obstacles
        for obstacle in self.obstacles:
            grid[obstacle[0]][obstacle[1]] = 'X'
        
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'
            
        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        #print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        #print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    taxi_row, taxi_col, _, _, _, _, _, _, _, _, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    done = False
    step_count = 0
    total_reward = 0

    if render:
        env.render_env((taxi_row, taxi_col), action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
        
    while not done:
        action = student_agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        print("reward:", reward, "total reward=", total_reward)
        step_count += 1
        taxi_row, taxi_col, _, _, _, _, _, _, _, _, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
        if render:
            env.render_env((taxi_row, taxi_col), action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {round(total_reward, 2)}")
    return total_reward

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid environment")
    parser.add_argument("--fuel_limit", type=int, default=5000, help="Maximum fuel available for the agent")
    parser.add_argument("--obstacles_percentage", type=float, default=0.1, help="Percentage of grid occupied by obstacles (0.0 to 1.0)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    env_config = {
        "grid_size": args.grid_size,
        "fuel_limit": args.fuel_limit,
        "obstacles_percentage": args.obstacles_percentage
    }
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {round(agent_score, 2)}")
    