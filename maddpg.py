import torch
import torch.optim as optim
import numpy as np
from networks import Actor, Critic
from replay_buffer import SharedReplayBuffer
from typing import Dict


class GaussianNoise:
    """
    Gaussian Noise generator for exploration during training.
    This noise is added to the agent's actions to encourage exploration.
    """
    def __init__(self, action_dim, mu=0.0, sigma_initial=0.1):
        """
        Initialize the Gaussian Noise parameters.

        Args:
            action_dim (int): Dimensionality of the action space.
            mu (float): Mean of the noise.
            sigma_initial (float): Initial standard deviation of the noise.
        """
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma_initial
        self.step = 0

    def reset(self):
        """Reset the noise generator. This is a placeholder method for future use."""
        pass

    def sample(self):
        """
        Sample noise for action perturbation based on decay schedule.

        Returns:
            np.ndarray: Sampled noise vector.
        """

        return np.random.normal(self.mu, self.sigma, self.action_dim)


class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for cooperative multi-agent environments.
    Implements both the actor-critic architecture and the target network for each agent type (predator, prey).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.95, tau=0.01, buffer_size=500000, batch_size=256, device=None):
        """
        Initialize the MADDPG parameters and networks.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of hidden units in the actor and critic networks.
            actor_lr (float): Learning rate for the actor networks.
            critic_lr (float): Learning rate for the critic networks.
            gamma (float): Discount factor for future rewards.
            tau (float): Target network soft update rate.
            buffer_size (int): Size of the experience replay buffer.
            batch_size (int): Size of the batch to sample from the replay buffer.
            device (torch.device): Device to run the model on (GPU/CPU).
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize the shared replay buffer
        self.replay_buffer = SharedReplayBuffer(buffer_size, batch_size, state_dim, action_dim)

        # Initialize Actor and Critic networks for both predator and prey agents
        self.predator_actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.prey_actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.predator_critic = Critic((state_dim + action_dim), hidden_dim).to(self.device)
        self.prey_critic = Critic((state_dim + action_dim), hidden_dim).to(self.device)

        # Initialize Target networks (for both actor and critic)
        self.predator_actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.prey_actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.predator_critic_target = Critic((state_dim + action_dim), hidden_dim).to(self.device)
        self.prey_critic_target = Critic((state_dim + action_dim), hidden_dim).to(self.device)

        # Copy weights from the initial actor and critic networks to target networks
        self.predator_actor_target.load_state_dict(self.predator_actor.state_dict())
        self.prey_actor_target.load_state_dict(self.prey_actor.state_dict())
        self.predator_critic_target.load_state_dict(self.predator_critic.state_dict())
        self.prey_critic_target.load_state_dict(self.prey_critic.state_dict())

        # Initialize optimizers for both actor and critic networks
        self.predator_actor_optimizer = optim.Adam(self.predator_actor.parameters(), lr=actor_lr)
        self.prey_actor_optimizer = optim.Adam(self.prey_actor.parameters(), lr=actor_lr)
        self.predator_critic_optimizer = optim.Adam(self.predator_critic.parameters(), lr=critic_lr)
        self.prey_critic_optimizer = optim.Adam(self.prey_critic.parameters(), lr=critic_lr)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Gaussian noise for exploration (for both predator and prey)
        self.predator_noise = GaussianNoise(action_dim)
        self.prey_noise = GaussianNoise(action_dim)

        # Normalization statistics for states (for both predator and prey)
        self.predator_state_mean = torch.zeros(state_dim).to(self.device)
        self.predator_state_std = torch.ones(state_dim).to(self.device)
        self.prey_state_mean = torch.zeros(state_dim).to(self.device)
        self.prey_state_std = torch.ones(state_dim).to(self.device)

    def select_action(self, states: Dict[str, np.ndarray], agent_type: str, add_noise=True) -> Dict[str, np.ndarray]:
        """
        Select actions for each agent using their respective actor networks.

        Args:
            states (dict): The state for each agent.
            agent_type (str): The type of agent ('predator' or 'prey').
            add_noise (bool): Whether to add exploration noise to the actions.

        Returns:
            dict: The selected actions for each agent.
        """
        if agent_type == 'predator':
            # states = {1:np.darray,2:np.darray}
            actions = {}
            for agent, state in states.items():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                normalized_state = (state_tensor - self.predator_state_mean) / (self.predator_state_std + 1e-8)

                with torch.no_grad():
                    action_tensor = self.predator_actor(normalized_state.unsqueeze(0))

                action = action_tensor.cpu().data.numpy().flatten()
                if add_noise:
                    action += self.predator_noise.sample()
                    action = np.clip(action, -1.0, 1.0)
                actions[agent] = action

            return actions

        elif agent_type == 'prey':
            actions = {}
            for agent, state in states.items():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                normalized_state = (state_tensor - self.prey_state_mean) / (self.prey_state_std + 1e-8)

                with torch.no_grad():
                    action_tensor = self.prey_actor(normalized_state.unsqueeze(0))

                action = action_tensor.cpu().data.numpy().flatten()
                if add_noise:
                    action += self.prey_noise.sample()
                    action = np.clip(action, -1.0, 1.0)

                actions[agent] = action

            return actions
        else:
            raise ValueError("Invalid agent type. Must be 'predator' or 'prey'")


    def train_step(self, agent_type: str):
        """
        Perform a single training step for the specified agent type (predator or prey).

        Args:
            agent_type (str): The type of agent ('predator' or 'prey').
        """
        if agent_type == 'predator':
            critic = self.predator_critic
            critic_target = self.predator_critic_target
            critic_optimizer = self.predator_critic_optimizer
            actor = self.predator_actor
            actor_target = self.predator_actor_target
            actor_optimizer = self.predator_actor_optimizer
        elif agent_type == 'prey':
            critic = self.prey_critic
            critic_target = self.prey_critic_target
            critic_optimizer = self.prey_critic_optimizer
            actor = self.prey_actor
            actor_target = self.prey_actor_target
            actor_optimizer = self.prey_actor_optimizer
        else:
            raise ValueError("Invalid agent type. Must be 'predator' or 'prey'.")

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(agent_type)

        # Normalize states
        self.update_normalization_stats(agent_type, states)

        if agent_type == 'predator':
            normalized_states = (states - self.predator_state_mean) / (self.predator_state_std + 1e-8)
            normalized_next_states = (next_states - self.predator_state_mean) / (self.predator_state_std + 1e-8)
        else:
            normalized_states = (states - self.prey_state_mean) / (self.prey_state_std + 1e-8)
            normalized_next_states = (next_states - self.prey_state_mean) / (self.prey_state_std + 1e-8)

        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

        # Critic update
        with torch.no_grad():
            target_actions = actor_target(normalized_next_states)

            target_q = rewards + self.gamma * critic_target(
                torch.cat([normalized_next_states, target_actions], dim=1)) * (1 - dones)


        current_q = critic(torch.cat([normalized_states, actions], dim=1))


        critic_loss = torch.nn.MSELoss()(current_q, target_q)

        # Update Critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        critic_optimizer.step()

        # Actor update
        current_actions = actor(normalized_states)
        actor_loss = -critic(torch.cat([normalized_states, current_actions], dim=1)).mean()

        # Update Actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_optimizer.step()

        # Soft update of target networks
        self.soft_update(critic_target, critic)
        self.soft_update(actor_target, actor)

    def update_normalization_stats(self, agent_type: str, states: torch.Tensor):
        """
        Update the normalization statistics (mean and std) for states.

        Args:
            agent_type (str): The type of agent ('predator' or 'prey').
            states (torch.Tensor): The batch of states.
        """
        if agent_type == 'predator':
            self.predator_state_mean = states.mean(dim=0)
            self.predator_state_std = states.std(dim=0)
        elif agent_type == 'prey':
            self.prey_state_mean = states.mean(dim=0)
            self.prey_state_std = states.std(dim=0)
        else:
            raise ValueError("Invalid agent type. Must be 'predator' or 'prey'.")

    def soft_update(self, target, source):
        """
        Perform a soft update of the target network using the source network.

        Args:
            target (torch.nn.Module): The target network.
            source (torch.nn.Module): The source network.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
