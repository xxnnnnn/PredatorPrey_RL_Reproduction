import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64, hidden_layer_num: int = 3):
        """
        Initializes the Actor network.

        :param input_dim: Dimension of the input state.
        :param action_dim: Dimension of the action space.
        :param hidden_dim: Number of hidden units in each hidden layer.
        :param hidden_layer_num: Number of hidden layers.
        """
        super(Actor, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layer_num - 1)]
        )
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Actor network.

        :param state: Input state.
        :return: Action to be taken, scaled between -1 and 1.
        """
        x = self.relu(self.fc1(state))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        action = torch.tanh(self.fc_out(x))
        return action


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, hidden_layer_num: int = 3):
        """
        Initializes the Critic network.

        :param input_dim: Dimension of the input, which includes state and action.
        :param hidden_dim: Number of hidden units in each hidden layer.
        :param hidden_layer_num: Number of hidden layers.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layer_num - 1)]
        )
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Critic network.

        :param state_action: Concatenated state and action tensor.
        :return: Q-value for the given state-action pair.
        """
        x = self.relu(self.fc1(state_action))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        q_value = self.fc_out(x)
        return q_value
