import os
import yaml
import torch
import gymnasium as gym

from dqn import DQN
from experience_replay import ReplayMemory
import flappy_bird_gymnasium
from torch import nn


class Agent:
    """
    Deep Q-Learning Agent that handles hyperparameters, networks, and optimization.
    """
    
    def __init__(self, hyperparameter_set):
        """
        Initialize the agent with specified hyperparameters.
        
        Args:
            hyperparameter_set (str): Name of hyperparameter set in hyperparameters.yml
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load hyperparameters
        self._load_hyperparameters(hyperparameter_set)
        
        # Initialize networks and optimizer (will be set up when needed)
        self.policy_dqn = None
        self.target_dqn = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        
        # Initialize environment info (will be set up when creating environment)
        self.env = None
        self.num_states = None
        self.num_actions = None
    
    def _load_hyperparameters(self, hyperparameter_set):
        """Load hyperparameters from YAML file."""
        try:
            with open('hyperparameters.yml', 'r') as file:
                all_hyperparameter_sets = yaml.safe_load(file)
                hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        except FileNotFoundError:
            raise FileNotFoundError("hyperparameters.yml file not found")
        except KeyError:
            raise KeyError(f"Hyperparameter set '{hyperparameter_set}' not found in hyperparameters.yml")
        
        self.hyperparameter_set = hyperparameter_set
        
        ## LOAD HYPERPARAMETERS
        
        # Environment parameters
        self.env_id = hyperparameters['env_id']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        
        # Learning parameters
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        
        # Memory and batch parameters
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        
        # Exploration parameters
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        
        # Training parameters
        self.stop_on_reward = hyperparameters['stop_on_reward']
        
        # Network architecture parameters
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        
        # File paths
        runs_dir = "runs"
        os.makedirs(runs_dir, exist_ok=True)
        self.log_file = os.path.join(runs_dir, f'{self.hyperparameter_set}.log')
        self.model_file = os.path.join(runs_dir, f'{self.hyperparameter_set}.pt')
        self.graph_file = os.path.join(runs_dir, f'{self.hyperparameter_set}.png')
    
    def setup_environment(self, render=False):
        """
        Create and setup the environment.
        
        Args:
            render (bool): Whether to render the environment
        """
        render_mode = 'human' if render else None
        self.env = gym.make(self.env_id, render_mode=render_mode, **self.env_make_params)
        
        # Get environment dimensions
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]
        
        print(f"Environment: {self.env_id}")
        print(f"State dimension: {self.num_states}")
        print(f"Action dimension: {self.num_actions}")
    
    def setup_networks(self, for_training=True):
        """
        Setup policy and target networks.
        
        Args:
            for_training (bool): Whether networks are being setup for training
        """
        if self.num_states is None or self.num_actions is None:
            raise ValueError("Environment must be setup before networks")
        
        # Create policy network
        self.policy_dqn = DQN(
            self.num_states, 
            self.num_actions, 
            self.fc1_nodes, 
            self.enable_dueling_dqn
        ).to(self.device)
        
        if for_training:
            # Create target network (copy of policy network)
            self.target_dqn = DQN(
                self.num_states, 
                self.num_actions, 
                self.fc1_nodes, 
                self.enable_dueling_dqn
            ).to(self.device)
            
            # Make target network identical to policy network
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            
            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.policy_dqn.parameters(), 
                lr=self.learning_rate_a
            )
            
            print("Networks setup for training")
        else:
            # Load trained model
            if not os.path.exists(self.model_file):
                raise FileNotFoundError(f"Model file not found: {self.model_file}")
            
            self.policy_dqn.load_state_dict(torch.load(self.model_file, map_location=self.device))
            self.policy_dqn.eval()
            
            print("Model loaded for evaluation")
    
    def setup_memory(self):
        """Setup experience replay memory."""
        return ReplayMemory(self.replay_memory_size)
    
    def select_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): Current state
            epsilon (float): Exploration probability
            
        Returns:
            torch.Tensor: Selected action as tensor
        """
        if torch.rand(1).item() < epsilon:
            # Random action
            action = self.env.action_space.sample()
            return torch.tensor(action, dtype=torch.int64, device=self.device)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_dqn(state.unsqueeze(dim=0)).squeeze()
                action = q_values.argmax()
                return action
    
    def optimize(self, mini_batch):
        """
        Optimize the policy network using a mini-batch of experiences.
        
        Args:
            mini_batch (list): List of (state, action, next_state, reward, done) tuples
        """
        if self.optimizer is None or self.target_dqn is None:
            raise ValueError("Networks must be setup for training before optimization")
        
        # Transpose and stack the mini-batch
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=self.device)
        
        # Calculate target Q-values
        with torch.no_grad():
            if self.enable_double_dqn:
                # Double DQN: use policy network to select actions, target network to evaluate
                best_actions = self.policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                          self.target_dqn(new_states).gather(dim=1, index=best_actions.unsqueeze(dim=1)).squeeze()
            else:
                # Standard DQN: use target network for both selection and evaluation
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                          self.target_dqn(new_states).max(dim=1)[0]
        
        # Calculate current Q-values
        current_q = self.policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        
        # Compute loss and optimize
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def sync_target_network(self):
        """Copy policy network parameters to target network."""
        if self.target_dqn is None:
            raise ValueError("Target network not initialized")
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
    
    def save_model(self):
        """Save the current policy network."""
        if self.policy_dqn is None:
            raise ValueError("Policy network not initialized")
        torch.save(self.policy_dqn.state_dict(), self.model_file)
    
    def get_state_dict(self):
        """Get the current policy network state dict."""
        if self.policy_dqn is None:
            return None
        return self.policy_dqn.state_dict()
    
    def cleanup(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()


# Test the agent if run directly
if __name__ == '__main__':
    # This is just for testing - create a simple hyperparameter set
    test_hyperparams = {
        'test': {
            'env_id': 'CartPole-v1',
            'learning_rate_a': 0.001,
            'discount_factor_g': 0.99,
            'network_sync_rate': 100,
            'replay_memory_size': 10000,
            'mini_batch_size': 32,
            'epsilon_init': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'stop_on_reward': 500,
            'fc1_nodes': 256,
            'enable_double_dqn': True,
            'enable_dueling_dqn': True
        }
    }
    
    # Create temporary hyperparameters file
    with open('hyperparameters.yml', 'w') as f:
        yaml.dump(test_hyperparams, f)
    
    try:
        agent = Agent('test')
        agent.setup_environment()
        agent.setup_networks(for_training=True)
        print("Agent test completed successfully!")
    except Exception as e:
        print(f"Agent test failed: {e}")
    finally:
        # Clean up
        if os.path.exists('hyperparameters.yml'):
            os.remove('hyperparameters.yml')