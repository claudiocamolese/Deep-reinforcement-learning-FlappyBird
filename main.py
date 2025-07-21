import argparse
import sys
import os
from datetime import datetime

from agent import Agent
from train import train_agent
from val import validate_agent

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    """Main function to parse arguments and execute training or validation."""
    parser = argparse.ArgumentParser(description='Train or test DQN model.')
    parser.add_argument('hyperparameters', help='Hyperparameter set name from hyperparameters.yml')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--val', help='Validation mode (same as not using --train)', action='store_true')
    parser.add_argument('--render', help='Render environment during validation', action='store_true')
    
    args = parser.parse_args()
    
    # Check arguments
    if args.train and args.val:
        print("Error: Cannot use both --train and --val flags simultaneously")
        sys.exit(1)
    
    # Create agent
    try:
        agent = Agent(hyperparameter_set=args.hyperparameters)
        print(f"Created agent with hyperparameter set: {args.hyperparameters}")
    except Exception as e:
        print(f"Error creating agent: {e}")
        sys.exit(1)
    
    # Execute training or validation
    if args.train:
        print(f"{datetime.now().strftime(DATE_FORMAT)}: Starting training...")
        train_agent(agent)
        print(f"{datetime.now().strftime(DATE_FORMAT)}: Ended training")
    else:
        print(f"{datetime.now().strftime(DATE_FORMAT)}: Starting validation...")
        validate_agent(agent, render=args.render)
        print(f"{datetime.now().strftime(DATE_FORMAT)}: Ended validation")


if __name__ == '__main__':
    main()