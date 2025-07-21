# Deep Q-Learning (DQN)

Deep Q-Learning, also known as Deep Q-Network (DQN), is an extension of the classic Q-Learning algorithm that uses deep neural networks to estimate Q-values. While traditional Q-Learning relies on a Q-table and works fine with small, discrete state spaces, it doesn’t scale well when states are large or continuous. DQN solves this by replacing the Q-table with a neural network that predicts Q-values for all possible actions given a state.

---

## Main Ideas Behind Deep Q-Learning

### Neural Network for Q-Value Approximation  
Instead of storing Q-values in a table, DQN uses a neural network which takes the state as input and outputs Q-values for every possible action. This lets us handle more complex or high-dimensional environments.

### Experience Replay  
To make training more stable and efficient, DQN keeps a replay buffer where it stores past experiences `(state, action, reward, next_state)`. When training, it samples random batches from this buffer, which helps reduce correlations between samples.

### Target Network  
DQN uses a separate target network, which is just a copy of the main network updated less frequently. This helps produce more stable target Q-values during training and avoids the learning process from oscillating too much.

### Bellman Equation Update  
The network is trained by minimizing the difference between predicted Q-values and targets computed using the Bellman equation: 

$$
Q(s,a)\gets Q(s,a) + \alpha(r+ \gamma \underset{a'}{max} Q(s', a', \theta^-) - Q(s,a; \theta))
$$

### Dueling Deep Q-Network (Dueling DQN)

Dueling DQN is an improvement over the standard DQN architecture that separately estimates the **state value** and the **advantage** for each action. Instead of directly outputting Q-values for each action, the network is split into two streams: one calculates the value of being in a particular state (V(s)), and the other estimates the advantage of each action in that state (A(s,a)). These two streams are then combined to produce the final Q-values. This separation helps the network learn which states are valuable without needing to learn the effect of each action in every state, often leading to faster and more stable learning, especially in environments where some actions do not affect the outcome much.

## How to run
Clone the repository with and install libraries.
```bash
git clone https://github.com/claudiocamolese/Deep-reinforcement-learning-FlappyBird.git
pip install -r requirements
```
The project includes a command-line interface to either train or validate the DQN agent. You need to specify a hyperparameter set defined in `hyperparameters.yml`.

### Usage
Change your hyperparameters as you want and then, in the terminal
```bash
python main.py <hyperparameter_set_name> [--train] [--val] [--render]
