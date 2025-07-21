"""
Experience Replay Memory implementation for DQN.
"""

import random
from collections import deque


class ReplayMemory:
    """
    Experience Replay Memory buffer for storing and sampling transitions.
    
    This implementation uses a deque with fixed maximum length for efficient
    memory management and random sampling of experiences for training.
    """
    
    def __init__(self, maxlen, seed=None):
        """
        Initialize the replay memory.
        
        Args:
            maxlen (int): Maximum number of transitions to store
            seed (int, optional): Random seed for reproducibility
        """
        self.memory = deque([], maxlen=maxlen)
        self.maxlen = maxlen
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
    
    def append(self, transition):
        """
        Add a transition to the memory.
        
        Args:
            transition (tuple): A tuple containing (state, action, next_state, reward, done)
        """
        self.memory.append(transition)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the memory.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            list: A list of sampled transitions
            
        Raises:
            ValueError: If batch_size is larger than the available memory
        """
        if batch_size > len(self.memory):
            raise ValueError(f"Cannot sample {batch_size} transitions from memory of size {len(self.memory)}")
        
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """
        Get the current size of the memory.
        
        Returns:
            int: Number of transitions currently stored
        """
        return len(self.memory)
    
    def is_full(self):
        """
        Check if the memory buffer is full.
        
        Returns:
            bool: True if memory is at maximum capacity
        """
        return len(self.memory) == self.maxlen
    
    def clear(self):
        """Clear all stored transitions from memory."""
        self.memory.clear()
    
    def get_all_transitions(self):
        """
        Get all stored transitions.
        
        Returns:
            list: All transitions in memory
        """
        return list(self.memory)
    
    def get_memory_usage(self):
        """
        Get memory usage statistics.
        
        Returns:
            dict: Dictionary containing memory statistics
        """
        return {
            'current_size': len(self.memory),
            'max_size': self.maxlen,
            'usage_percentage': (len(self.memory) / self.maxlen) * 100,
            'is_full': self.is_full()
        }


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay Memory (optional advanced implementation).
    
    This version implements prioritized sampling where transitions with higher
    temporal difference (TD) errors are sampled more frequently.
    """
    
    def __init__(self, maxlen, alpha=0.6, beta=0.4, beta_increment=0.001, seed=None):
        """
        Initialize the prioritized replay memory.
        
        Args:
            maxlen (int): Maximum number of transitions to store
            alpha (float): Prioritization exponent (0 = uniform sampling, 1 = full prioritization)
            beta (float): Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment (float): Beta increment per sampling step
            seed (int, optional): Random seed for reproducibility
        """
        self.maxlen = maxlen
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Storage
        self.memory = deque([], maxlen=maxlen)
        self.priorities = deque([], maxlen=maxlen)
        
        if seed is not None:
            random.seed(seed)
    
    def append(self, transition, td_error=None):
        """
        Add a transition to the memory with priority.
        
        Args:
            transition (tuple): A tuple containing (state, action, next_state, reward, done)
            td_error (float, optional): Temporal difference error for prioritization
        """
        # Use max priority for new transitions if no TD error provided
        priority = self.max_priority if td_error is None else abs(td_error) + 1e-6
        
        self.memory.append(transition)
        self.priorities.append(priority)
        
        # Update max priority
        if priority > self.max_priority:
            self.max_priority = priority
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions using prioritized sampling.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: (transitions, indices, weights) where:
                - transitions: List of sampled transitions
                - indices: Indices of sampled transitions
                - weights: Importance sampling weights
        """
        if batch_size > len(self.memory):
            raise ValueError(f"Cannot sample {batch_size} transitions from memory of size {len(self.memory)}")
        
        # Calculate sampling probabilities
        priorities_array = list(self.priorities)
        priorities_pow = [p ** self.alpha for p in priorities_array]
        total_priority = sum(priorities_pow)
        
        probabilities = [p / total_priority for p in priorities_pow]
        
        # Sample indices based on probabilities
        indices = random.choices(range(len(self.memory)), weights=probabilities, k=batch_size)
        
        # Get transitions and calculate importance sampling weights
        transitions = [self.memory[i] for i in indices]
        
        # Calculate weights for importance sampling correction
        weights = []
        min_prob = min(probabilities)
        for i in indices:
            prob = probabilities[i]
            weight = (len(self.memory) * prob) ** (-self.beta)
            weights.append(weight)
        
        # Normalize weights
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return transitions, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities for specific transitions.
        
        Args:
            indices (list): Indices of transitions to update
            td_errors (list): New temporal difference errors
        """
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                priority = abs(td_error) + 1e-6
                self.priorities[idx] = priority
                
                if priority > self.max_priority:
                    self.max_priority = priority
    
    def __len__(self):
        """Get the current size of the memory."""
        return len(self.memory)


# Test the replay memory if run directly
if __name__ == '__main__':
    import torch
    import numpy as np
    
    print("Testing Basic Replay Memory...")
    
    # Test basic functionality
    memory = ReplayMemory(maxlen=1000, seed=42)
    
    # Add some dummy transitions
    for i in range(10):
        state = torch.randn(4)
        action = torch.randint(0, 2, (1,))
        next_state = torch.randn(4)
        reward = torch.randn(1)
        done = i % 3 == 0  # Some episodes end
        
        transition = (state, action, next_state, reward, done)
        memory.append(transition)
    
    print(f"Memory size: {len(memory)}")
    print(f"Memory usage: {memory.get_memory_usage()}")
    
    # Test sampling
    if len(memory) >= 5:
        batch = memory.sample(5)
        print(f"Sampled batch size: {len(batch)}")
        print("Sample transition structure:")
        state, action, next_state, reward, done = batch[0]
        print(f"  State shape: {state.shape}")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
    
    print("\nTesting Prioritized Replay Memory...")
    
    # Test prioritized memory
    priority_memory = PrioritizedReplayMemory(maxlen=1000, seed=42)
    
    # Add transitions with different priorities
    for i in range(10):
        state = torch.randn(4)
        action = torch.randint(0, 2, (1,))
        next_state = torch.randn(4)
        reward = torch.randn(1)
        done = i % 3 == 0
        
        transition = (state, action, next_state, reward, done)
        td_error = np.random.random()  # Random TD error for testing
        priority_memory.append(transition, td_error)
    
    print(f"Priority memory size: {len(priority_memory)}")
    
    # Test prioritized sampling
    if len(priority_memory) >= 5:
        transitions, indices, weights = priority_memory.sample(5)
        print(f"Sampled batch size: {len(transitions)}")
        print(f"Sample indices: {indices}")
        print(f"Sample weights: {weights}")
        
        # Test priority updates
        new_td_errors = [0.5, 0.8, 0.1, 0.9, 0.3]
        priority_memory.update_priorities(indices, new_td_errors)
        print("Updated priorities successfully")
    
    print("\nReplay Memory tests completed!")