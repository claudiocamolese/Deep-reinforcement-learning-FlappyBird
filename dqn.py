import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network with optional Dueling architecture and Dropout.

    Args:
        state_dim (int): Dimension of the state space
        action_dim (int): Number of possible actions
        hidden_dim (int): Base number of neurons for hidden layers
        enable_dueling_dqn (bool): Whether to use dueling architecture
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor: 3 hidden layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.1)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(p=0.1)

        if self.enable_dueling_dqn:
            # Value stream: V(s)
            self.fc_value = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, 1)

            # Advantage stream: A(s,a)
            self.fc_advantages = nn.Linear(hidden_dim, hidden_dim)
            self.advantages = nn.Linear(hidden_dim, action_dim)
        else:
            # Standard DQN output
            self.fc_out1 = nn.Linear(hidden_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # Shared MLP with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        if self.enable_dueling_dqn:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            Q = V + A - A.mean(dim=1, keepdim=True)
        else:
            x = F.relu(self.fc_out1(x))
            Q = self.output(x)

        return Q

    def get_action(self, state, epsilon=0.0):
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return q_values.argmax().item()




# Test the network if run directly
if __name__ == '__main__':
    # Test both architectures
    state_dim = 12
    action_dim = 2
    batch_size = 10
    
    print("Testing Standard DQN...")
    net_standard = DQN(state_dim, action_dim, enable_dueling_dqn=False)
    state = torch.randn(batch_size, state_dim)
    output_standard = net_standard(state)
    print(f"Input shape: {state.shape}")
    print(f"Output shape: {output_standard.shape}")
    print(f"Sample output: {output_standard[0]}")
    
    print("\nTesting Dueling DQN...")
    net_dueling = DQN(state_dim, action_dim, enable_dueling_dqn=True)
    output_dueling = net_dueling(state)
    print(f"Input shape: {state.shape}")
    print(f"Output shape: {output_dueling.shape}")
    print(f"Sample output: {output_dueling[0]}")
    
    print("\nTesting action selection...")
    single_state = torch.randn(state_dim)
    action = net_dueling.get_action(single_state, epsilon=0.1)
    print(f"Selected action: {action}")