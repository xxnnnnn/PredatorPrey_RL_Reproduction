# replay_buffer.py
import numpy as np
import random
import torch
from typing import Tuple, Dict


class ReplayBuffer:
    """
    ReplayBuffer stores experiences for a specific agent type.
    Each experience includes the states and actions of all agents of that type.
    """

    def __init__(self, buffer_size: int, batch_size: int, state_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

    def add(self, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
            rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
            dones: Dict[str, bool]):
        """
        Adds a new experience to the buffer.
        Stores only experiences of agents of this buffer's type.
        """
        for agent_id in states.keys():
            state = states[agent_id]
            action = actions[agent_id]
            reward = rewards[agent_id]
            next_state = next_states[agent_id]
            done = dones[agent_id]
            # Create the experience tuple for this agent
            experience = (state, action, reward, next_state, done)

            # Add the experience to the buffer
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
                self.position = (self.position + 1) % self.buffer_size

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly samples a batch of experiences from the buffer and converts them to torch.Tensors.
        """
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将列表转换为单一的 numpy.ndarray
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)


        # Convert numpy arrays to torch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# SharedReplayBuffer for each agent type
class SharedReplayBuffer:
    """
    SharedReplayBuffer manages replay buffers for each agent type.
    Each type shares the same network and replay buffer.
    """

    def __init__(self, buffer_size: int, batch_size: int, state_dim: int, action_dim: int):
        """
        Initializes shared replay buffers for each agent type.

        :param buffer_size: Maximum number of experiences each buffer can hold.
        :param batch_size: Number of experiences to sample in each batch.
        :param state_dim: Dimension of each agent's state.
        :param action_dim: Dimension of each agent's action.
        """
        self.predator_buffer = ReplayBuffer(buffer_size, batch_size, state_dim, action_dim)
        self.prey_buffer = ReplayBuffer(buffer_size, batch_size, state_dim, action_dim)

    def add(self, agent_type: str, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
            rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
            dones: Dict[str, bool]):
        """
        Adds a new experience to the appropriate buffer based on agent type.
        """
        if agent_type == 'predator':
            self.predator_buffer.add(states, actions, rewards, next_states, dones)
        elif agent_type == 'prey':
            self.prey_buffer.add(states, actions, rewards, next_states, dones)
        else:
            raise ValueError("Invalid agent type. Must be 'predator' or 'prey'.")

    def sample(self, agent_type: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly samples a batch of experiences from the appropriate buffer based on agent type.
        """
        if agent_type == 'predator':
            return self.predator_buffer.sample()
        elif agent_type == 'prey':
            return self.prey_buffer.sample()
        else:
            raise ValueError("Invalid agent type. Must be 'predator' or 'prey'.")

    def buffer_length(self) -> Dict[str, int]:
        """
        Returns the current size of each internal buffer.
        """
        return {
            'predator': len(self.predator_buffer),
            'prey': len(self.prey_buffer)
        }
