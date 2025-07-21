import os
import torch
import random
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from experience_replay import ReplayMemory

# Use 'Agg' backend for matplotlib to generate plots without display
matplotlib.use('Agg')

# Date format for logging
DATE_FORMAT = "%m-%d %H:%M:%S"


def train_agent(agent):
    """
    Train the DQN agent using the specified hyperparameters.
    
    Args:
        agent (Agent): The agent to train
    """
    # Setup environment and networks for training
    agent.setup_environment(render=False)
    agent.setup_networks(for_training=True)
    
    # Initialize training variables
    start_time = datetime.now()
    last_graph_update_time = start_time
    
    # Logging
    log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
    print(log_message)
    with open(agent.log_file, 'w') as file:
        file.write(log_message + '\n')
    
    # Initialize training components
    epsilon = agent.epsilon_init
    memory = agent.setup_memory()
    
    # Tracking variables
    rewards_per_episode = []
    epsilon_history = []
    step_count = 0
    best_reward = -float('inf')
    
    # Training loop - runs indefinitely until manually stopped
    try:
        for episode in itertools.count():
            episode_reward = _run_episode(
                agent, memory, epsilon, step_count, episode
            )
            
            # Update tracking variables
            rewards_per_episode.append(episode_reward)
            step_count = _update_step_count(agent, memory, epsilon_history, step_count)
            
            # Update epsilon
            if len(memory) > agent.mini_batch_size:
                epsilon = max(epsilon * agent.epsilon_decay, agent.epsilon_min)
                epsilon_history.append(epsilon)
            
            # Save model if new best reward
            if episode_reward > best_reward:
                best_reward = _handle_new_best_reward(
                    agent, episode_reward, best_reward, episode
                )
            
            # Update graphs periodically
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                _save_training_graph(agent, rewards_per_episode, epsilon_history)
                last_graph_update_time = current_time
            
            # Log progress periodically
            if episode % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:]) if rewards_per_episode else 0
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}"
                print(log_message)
                with open(agent.log_file, 'a') as file:
                    file.write(log_message + '\n')
    
    except KeyboardInterrupt:
        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Training interrupted by user"
        print(log_message)
        with open(agent.log_file, 'a') as file:
            file.write(log_message + '\n')
    
    finally:
        # Save final results
        _save_training_graph(agent, rewards_per_episode, epsilon_history)
        
        # Final log
        end_time = datetime.now()
        duration = end_time - start_time
        log_message = f"{end_time.strftime(DATE_FORMAT)}: Training completed. Duration: {duration}, Episodes: {len(rewards_per_episode)}, Best Reward: {best_reward:.2f}"
        print(log_message)
        with open(agent.log_file, 'a') as file:
            file.write(log_message + '\n')
        
        # Cleanup
        agent.cleanup()


def _run_episode(agent, memory, epsilon, step_count, episode):
    """
    Run a single training episode.
    
    Args:
        agent (Agent): The agent
        memory (ReplayMemory): Experience replay memory
        epsilon (float): Current exploration rate
        step_count (int): Current step count
        episode (int): Current episode number
        
    Returns:
        float: Total reward for the episode
    """
    # Reset environment
    state, _ = agent.env.reset()
    state = torch.tensor(state, dtype=torch.float, device=agent.device)
    
    terminated = False
    episode_reward = 0.0
    
    # Run episode
    while not terminated and episode_reward < agent.stop_on_reward:
        # Select action
        action = agent.select_action(state, epsilon)
        
        # Execute action
        new_state, reward, terminated, truncated, info = agent.env.step(action.item())
        episode_reward += reward
        
        # Convert to tensors
        new_state = torch.tensor(new_state, dtype=torch.float, device=agent.device)
        reward = torch.tensor(reward, dtype=torch.float, device=agent.device)
        
        # Store experience
        memory.append((state, action, new_state, reward, terminated))
        
        # Move to next state
        state = new_state
        step_count += 1
    
    return episode_reward


def _update_step_count(agent, memory, epsilon_history, step_count):
    """
    Update networks and perform optimization if enough experience is available.
    
    Args:
        agent (Agent): The agent
        memory (ReplayMemory): Experience replay memory
        epsilon_history (list): History of epsilon values
        step_count (int): Current step count
        
    Returns:
        int: Updated step count
    """
    # Perform optimization if enough experience
    if len(memory) > agent.mini_batch_size:
        mini_batch = memory.sample(agent.mini_batch_size)
        loss = agent.optimize(mini_batch)
        
        # Sync target network periodically
        if step_count >= agent.network_sync_rate:
            agent.sync_target_network()
            step_count = 0
    
    return step_count


def _handle_new_best_reward(agent, episode_reward, best_reward, episode):
    """
    Handle new best reward by logging and saving model.
    
    Args:
        agent (Agent): The agent
        episode_reward (float): Current episode reward
        best_reward (float): Previous best reward
        episode (int): Current episode number
        
    Returns:
        float: New best reward
    """
    improvement = ((episode_reward - best_reward) / abs(best_reward) * 100) if best_reward != 0 else float('inf')
    
    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:.1f} ({improvement:+.1f}%) at episode {episode}, saving model..."
    print(log_message)
    with open(agent.log_file, 'a') as file:
        file.write(log_message + '\n')
    
    agent.save_model()
    return episode_reward


def _save_training_graph(agent, rewards_per_episode, epsilon_history):
    """
    Save training progress graphs.
    
    Args:
        agent (Agent): The agent
        rewards_per_episode (list): List of episode rewards
        epsilon_history (list): History of epsilon values
    """
    if not rewards_per_episode:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot mean rewards (rolling average over last 100 episodes)
    mean_rewards = np.zeros(len(rewards_per_episode))
    for i in range(len(mean_rewards)):
        start_idx = max(0, i - 99)
        mean_rewards[i] = np.mean(rewards_per_episode[start_idx:i+1])
    
    ax1.plot(mean_rewards, 'b-', linewidth=1)
    ax1.set_ylabel('Mean Rewards (100-episode average)')
    ax1.set_xlabel('Episodes')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Training Progress')
    
    # Plot epsilon decay
    if epsilon_history:
        ax2.plot(epsilon_history, 'r-', linewidth=1)
        ax2.set_ylabel('Epsilon')
        ax2.set_xlabel('Training Steps')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Exploration Decay')
    
    plt.tight_layout()
    plt.savefig(agent.graph_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


# If run directly, provide a simple test
if __name__ == '__main__':
    import yaml
    from agent import Agent
    
    # Create a simple test configuration
    test_config = {
        'test_train': {
            'env_id': 'CartPole-v1',
            'learning_rate_a': 0.001,
            'discount_factor_g': 0.99,
            'network_sync_rate': 100,
            'replay_memory_size': 1000,
            'mini_batch_size': 32,
            'epsilon_init': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'stop_on_reward': 200,
            'fc1_nodes': 128,
            'enable_double_dqn': True,
            'enable_dueling_dqn': True
        }
    }
    
    # Save test config
    with open('hyperparameters.yml', 'w') as f:
        yaml.dump(test_config, f)
    
    try:
        print("Starting training test...")
        agent = Agent('test_train')
        train_agent(agent)
    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        # Cleanup
        if os.path.exists('hyperparameters.yml'):
            os.remove('hyperparameters.yml')