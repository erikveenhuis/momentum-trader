"""Demo script that runs a random agent in the trading environment."""
import os
import gymnasium as gym
import argparse
from gymnasium.utils.env_checker import check_env
from momentum_env.config import TradingEnvConfig

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Run a random agent in the trading environment.')
    parser.add_argument('--skip-env-check', action='store_true', help='Skip the gymnasium environment check.')
    args = parser.parse_args()

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'tests', 'fixtures', '2017-06-27_BTC-USD.csv')
    
    # Register and create environment
    gym.register(id='momentum-env-v0', entry_point='momentum_env.environment:TradingEnv')
    env = gym.make('momentum-env-v0', 
        config=TradingEnvConfig(
            data_path=data_path,
            window_size=60,
            initial_balance=10000.0,
            transaction_fee=0.001,
            reward_scale=10.0,
            invalid_action_penalty=-1.0,
            render_mode='human'
        )
    )
    
    # Conditionally run check_env
    if not args.skip_env_check:
        print("Checking environment validity...")
        check_env(env.unwrapped)
        print("Environment check passed!")
    
    observation, info = env.reset()
    env.render()
    
    step = 0
    while True:
        step += 1
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        
        if not (terminated or truncated):
            env.render()
        
        if terminated or truncated:
            print(f"\nEpisode finished after {step} steps!")
            print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
            print(f"Total transaction cost: ${info['transaction_cost']:.2f}")
            break
    
    env.close()

if __name__ == "__main__":
    main() 