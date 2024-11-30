# predator_prey_env.py
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from typing import Dict, Tuple, Any
from gui import Gui  # Import the Gui class

# Type aliases
AgentID = str
ObsType = np.ndarray

class Env(ParallelEnv):
    def __init__(self):
        super().__init__()
        # Environment dimensions
        self.width = 150
        self.height = 150

        # Number of agents
        self.num_predators = 2
        self.num_preys = 7

        # Create agent list
        self.predators = [f'predator_{i}' for i in range(self.num_predators)]
        self.preys = [f'prey_{i}' for i in range(self.num_preys)]
        self.agents = self.predators + self.preys

        # Define action space range
        self.max_acceleration = 1
        self.max_angular_velocity = 0.5

        # Action space: linear acceleration and angular velocity
        self.action_spaces: Dict[AgentID, spaces.Space] = {
            agent: spaces.Box(
                low=np.array([0.0, -self.max_angular_velocity], dtype=np.float32),
                high=np.array([self.max_acceleration, self.max_angular_velocity], dtype=np.float32),
                dtype=np.float32
            )
            for agent in self.agents
        }

        # Observation space dimensions
        # own_state: position (2) + speed (1) + heading (1) = 4
        # neighbors: 6 preys * (relative_position (2) + heading (1)) +
        #            6 predators * (relative_position (2) + heading (1)) = 36
        # Total observation_dim = 40
        self.num_neighbors = 6
        self.neighbor_dim = 3  # relative_position (2) + heading (1)
        self.observation_dim = 4 + self.num_neighbors * self.neighbor_dim * 2  # 40

        self.observation_spaces: Dict[AgentID, spaces.Space] = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
            )
            for agent in self.agents
        }

        # Perception settings
        self.perception_radius = max(self.width,self.height)

        # Time step and physical parameters
        self.dt = 0.1  # Time step
        self.max_speed = 5.0  # Maximum speed of agents

        # Declare instance attributes
        self.agent_positions: Dict[AgentID, np.ndarray] = {}
        self.agent_speeds: Dict[AgentID, float] = {}
        self.agent_headings: Dict[AgentID, float] = {}
        self._dones: Dict[AgentID, bool] = {}  # Use _dones
        self.rewards: Dict[AgentID, float] = {}
        self.infos: Dict[AgentID, Dict] = {}
        self._step_count = 0
        self.epoch_counter = 0

        # Define agent radii
        self.predator_radius = 10.0  # Radius for predators
        self.prey_radius = 8.0      # Radius for preys
        self.agent_radii: Dict[AgentID, float] = {
            agent: self.predator_radius if 'predator' in agent else self.prey_radius
            for agent in self.agents
        }

        # Define maximum steps per episode
        self.max_steps = 4000

        # Monitor captured preys
        self.captured_preys: Dict[AgentID, bool] = {prey: False for prey in self.preys}

        # Initialize GUI
        self.gui = Gui(self.width, self.height)

        # Boundary crossing penalty
        self.boundary_penalty = -0.5  # Adjust penalty value as needed

        # Initialize agent states
        self.reset()

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[AgentID, ObsType], Dict[AgentID, Dict]]:

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize positions randomly or based on specified options
        if options and 'initial_positions' in options:
            initial_positions = options['initial_positions']
            self.agent_positions = {
                agent: np.array(initial_positions.get(agent, [
                    np.random.uniform(0, self.width),
                    np.random.uniform(0, self.height)
                ]), dtype=np.float32)
                for agent in self.agents
            }
        else:
            self.agent_positions = {
                agent: np.random.uniform(0, self.width, 2).astype(np.float32)
                for agent in self.agents
            }

        # Initialize speeds to zero
        self.agent_speeds = {
            agent: 0.0 for agent in self.agents
        }

        # Initialize headings randomly
        self.agent_headings = {
            agent: np.random.uniform(-np.pi , np.pi) for agent in self.agents
        }

        # Initialize done flags and rewards
        self._dones = {agent: False for agent in self.agents}  # Use _dones
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Record current step count
        self._step_count = 0

        # Reset captured preys
        self.captured_preys = {prey: False for prey in self.preys}

        # Return observations and info
        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def observe(self, agent: AgentID) -> ObsType:
        """
        Returns the observation for a given agent.
        Each agent observes:
        - Its own state: position (2), speed (1), heading (1)
        - Up to 6 closest preys and 6 closest predators within its perception radius.
        """
        # Own state: position (2) + speed (1) + heading (1)
        position = self.agent_positions[agent]
        speed = self.agent_speeds[agent]
        heading = self.agent_headings[agent]
        own_state = np.concatenate([position, [speed], [heading]])

        # Separate observed predators and preys
        observed_predators = []
        observed_preys = []

        for other_agent in self.agents:
            if other_agent == agent:
                continue  # Skip itself

            # Calculate distance without wrap-around
            dx = self.agent_positions[other_agent][0] - self.agent_positions[agent][0]
            dy = self.agent_positions[other_agent][1] - self.agent_positions[agent][1]

            distance = np.sqrt(dx ** 2 + dy ** 2)

            if distance <= self.perception_radius:
                # Relative position
                relative_position = np.array([dx, dy], dtype=np.float32)

                # Compute the heading of the observed agent
                other_heading = self.agent_headings[other_agent]

                # Neighbor information
                neighbor_info = np.concatenate([relative_position, [other_heading]])

                # Classify neighbor as predator or prey
                if "predator" in other_agent:
                    observed_predators.append((distance, neighbor_info))
                else:
                    observed_preys.append((distance, neighbor_info))

        # Sort neighbors by distance
        observed_predators = sorted(observed_predators, key=lambda x: x[0])
        observed_preys = sorted(observed_preys, key=lambda x: x[0])

        # Select up to 6 closest neighbors for each category
        observed_predators = [info[1] for info in observed_predators[:self.num_neighbors]]
        observed_preys = [info[1] for info in observed_preys[:self.num_neighbors]]

        # Pad with zeros if necessary
        if len(observed_predators) < self.num_neighbors:
            padding = [np.zeros(self.neighbor_dim, dtype=np.float32) for _ in
                       range(self.num_neighbors - len(observed_predators))]
            observed_predators.extend(padding)

        if len(observed_preys) < self.num_neighbors:
            padding = [np.zeros(self.neighbor_dim, dtype=np.float32) for _ in
                       range(self.num_neighbors - len(observed_preys))]
            observed_preys.extend(padding)

        # Flatten neighbor information
        predators_flat = np.concatenate(observed_predators).astype(np.float32)
        preys_flat = np.concatenate(observed_preys).astype(np.float32)

        # Final observation: own_state + predators + preys
        observation = np.concatenate([own_state, predators_flat, preys_flat])

        return observation

    def step(self, actions: Dict[AgentID, np.ndarray]) -> Tuple[
        Dict[AgentID, ObsType], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, bool], Dict[AgentID, Dict]]:
        """
        Executes one step in the environment, updating the state based on agents' actions,
        calculating rewards, and determining if the episode has ended.

        :param actions: A dictionary mapping agent IDs to their actions (linear acceleration and angular velocity).
        :return: A tuple containing observations, rewards, terminated flags, truncated flags, and info dictionaries.
        """
        # Increment step count
        self._step_count += 1

        # Reset all agents' rewards
        self.rewards = {agent: 0.0 for agent in self.agents}

        # Record current state before applying actions
        current_states = {agent: self.observe(agent) for agent in self.agents if not self._dones[agent]}

        # Apply actions and update agent states (position, speed, heading)
        for agent, action in actions.items():
            if not self._dones[agent]:
                # Parse action: linear acceleration and angular velocity
                linear_acceleration, angular_velocity = action

                # Update heading
                self.agent_headings[agent] += 0.5 * angular_velocity * self.dt
                self.agent_headings[agent] %= 2 * np.pi  # Keep heading within [0, 2π)

                # Update speed with acceleration
                new_speed = self.agent_speeds[agent] + 100 * linear_acceleration * self.dt
                # Limit speed to max_speed
                new_speed = np.clip(new_speed, 0.0, self.max_speed)
                self.agent_speeds[agent] = new_speed

                # Update position based on speed and heading
                velocity = self.agent_speeds[agent] * np.array([
                    np.cos(self.agent_headings[agent]),
                    np.sin(self.agent_headings[agent])
                ], dtype=np.float32)
                new_position = self.agent_positions[agent] + velocity * self.dt

                # Check for boundary crossing
                crossed_boundary = False
                if new_position[0] < 0 or new_position[0] > self.width:
                    crossed_boundary = True
                if new_position[1] < 0 or new_position[1] > self.height:
                    crossed_boundary = True

                if crossed_boundary:
                    # Apply boundary crossing penalty
                    self.rewards[agent] += self.boundary_penalty

                # Wrap position around the environment boundaries
                new_position[0] %= self.width
                new_position[1] %= self.height
                self.agent_positions[agent] = new_position

                # Calculate energy consumption penalty
                penalty = -0.01 * abs(linear_acceleration) - 0.1 * abs(angular_velocity)
                self.rewards[agent] += penalty  # Add penalty to reward

        # Handle capture behavior: predators capturing preys
        for prey in self.preys:
            being_captured = False  # Flag to indicate if the prey is being captured

            for predator in self.predators:
                # Calculate distance between predator and prey without wrap-around
                dx = self.agent_positions[predator][0] - self.agent_positions[prey][0]
                dy = self.agent_positions[predator][1] - self.agent_positions[prey][1]

                distance = np.sqrt(dx ** 2 + dy ** 2)

                # Check if within capture distance (sum of radii)
                if distance < (self.agent_radii[predator] + self.agent_radii[prey]):
                    being_captured = True
                    # Predator receives positive reward
                    self.rewards[predator] += 1

            if being_captured:
                # Prey is being captured; receive negative reward
                self.rewards[prey] -= 1
                self.captured_preys[prey] = True
            else:
                if self.captured_preys[prey]:
                    # Prey is no longer being captured; reset reward
                    self.rewards[prey] = 0.0
                    self.captured_preys[prey] = False

        # Record next state after actions have been applied
        next_states = {agent: self.observe(agent) for agent in self.agents if not self._dones[agent]}

        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}  # Continuous environments可能设置为False

        # Prepare info dictionaries
        infos = {agent: {} for agent in self.agents}

        # Prepare observations (only for agents that are not done)
        observations = {agent: next_states[agent] for agent in next_states}

        return observations, self.rewards, terminated, truncated, infos

    def render(self, mode="rgb_array"):
        self.gui.handle_events()
        self.gui.update_display(self.agent_positions, self.agent_headings, self.epoch_counter, self.agent_radii)
        import pygame.surfarray
        rgb_array = pygame.surfarray.array3d(self.gui.screen)
        return np.flipud(np.rot90(rgb_array))

    def render_eval(self, mode="rgb_array"):
        self.gui.handle_events()
        self.gui.update_display(self.agent_positions, self.agent_headings, self.epoch_counter, self.agent_radii, eval=True)
        import pygame.surfarray
        rgb_array = pygame.surfarray.array3d(self.gui.screen)
        return np.flipud(np.rot90(rgb_array))

    def close(self):
        """
        Closes the GUI and performs necessary cleanup.
        """
        if hasattr(self, 'gui') and self.gui is not None:
            self.gui.close()
            self.gui = None
        super().close()

    @property
    def dones(self) -> Dict[AgentID, bool]:
        return self._dones