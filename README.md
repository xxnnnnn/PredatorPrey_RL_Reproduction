# PredatorPrey_RL_Reproduction

This project is a reproduction of the research conducted by Shiyu Zhao, a researcher at Westlake University. Below are the details of the original article:

**Li, J., Li, L., & Zhao, S.** (2023). "Predator–prey survival pressure is sufficient to evolve swarming behaviors." *New Journal of Physics*, 25(9), 092001. 

The article can be accessed here: https://iopscience.iop.org/article/10.1088/1367-2630/acf33a

Additionally,there are also some videos that discuss concepts similar to those presented in the article.The video links are listed below.

- [Evolving AIs - Predator vs Prey, who will win?](https://www.youtube.com/watch?v=qwrp3lB-jkQ&t=592s)
- [Much bigger simulation, AIs learn Phalanx](https://www.youtube.com/watch?v=tVNoetVLuQg&t=815s)

## Article overview
The main idea of this article is that when using reinforcement learning (RL) to simulate swarm behavior, specific reward functions may cause the swarm to exhibit certain behaviors. However, such tailored rewards are not necessary. Instead, the coevolution of the swarm can be achieved using the simplest form of reward—**the pressure of survival**.

## Code Description
PredatorPrey_RL_Reproduction/

├── gui.py                  # Code for the graphical user interface.

├── maddpg.py               # Implementation of the MADDPG algorithm used in the project.

├── train.py                 # Training the model in the project.

├── networks.py             # Defines the actor-critic network, shared among agents of the same type.

├── predator_prey_env.py    # Implementation of the predator-prey simulation environment.

├── replay_buffer.py        # Replay buffer shared by agents of the same type for experience replay.

├── eval.py        # Eval the model in the project.

**The following sections of the file are encouraged to be modified as outlined below:**

- **gui.py**
  - **Grid**: Customize light/dark grid spacing and colors for visualization.
  - **Agents**: Modify the appearance of predators (red) and prey (green), including their headings and sizes.
  - **Display Updates**: Adjust real-time updates of the grid, agent positions, and epoch information.
  - **Frame Rate**: Configure FPS for smoother or faster visualization.

- **maddpg.py**
  - **Gaussian Noise**: Modify the noise parameters to adjust exploration during training.
  - **Exploration**: The current code lacks explicit exploration functionality, which can make training challenging in early episodes.

- **train.py**
  - **Video Recording**:
    - Adjust the settings for saving episode videos, such as frame dimensions and encoding requirements.

- **predator_prey_env.py**
  - **Environment Features**:
    - **State Dynamics**:
      - Add functionality to avoid collisions for more realistic behavior.
    - **Rewards**:
      - Modify reward functions to explore different behavioral outcomes.
  - **Key Components**:
    - **Perception**: Adjust the perception range to observe various phenomena, as suggested in the referenced article.

- **eval.py**
  - **Video Generation**:
    - Modify the FPS, video length and anything else to tailor the evaluation output.

It is worth noting that there are some simplifications and omissions in my code. However, these modifications are unrelated to the core content of the article.

For example, the article prohibits agents from colliding by using Hooke's law, but this aspect has been omitted in the code. Additionally, the output of the actor network has been scaled in the code to ensure smoother trajectories during simulation. 

These changes do not affect the fundamental essence of the article.


## Further Information on the Article
### Homogeneity
Since the game involves only two species—prey and predator—it is intuitive that agents within the same species share identical strategies, as they belong to the same ecological role. However, individual agents may perform distinct actions based on differences in their observations. To simplify the structure, agents of the same species utilize a shared Actor-Critic network and replay buffer, reducing redundancy while maintaining consistency in decision-making.

### Neural Networks and Replay Buffer
This article implements the Actor-Critic network, where the actor network selects the agent's action based on its current observation. The actor's outputs have two dimensions:  **a<sub>F</sub> (angular acceleration)** and  **a<sub>R</sub> (radial acceleration)**. The critic network evaluates the expected future reward corresponding to the chosen action, enabling the agent to optimize its strategy over time.

The replay buffer plays a crucial role in enabling the shared learning architecture. For each conspecies, a single replay buffer is maintained, ensuring uniform data collection and utilization across all agents. During training, experiences from all agents of the same species are aggregated into the shared buffer. This collective learning framework allows agents to benefit from the experiences of their peers, leading to more robust and generalized strategies.

The network consists of three hidden layers, each with 64 units. The **Critic Network** outputs a single value representing the expected reward, while the **Actor Network** outputs two values: angular acceleration **a<sub>F</sub>** and radial acceleration  **a<sub>R</sub>**.
 

### The Hyper-parameter
| **Hyper-parameter**         | **Value**  |
|-----------------------------|------------|
| Number of episodes          | 2000(8500 in code)      |
| Episode length              | 100 (400 in code)       |
| Number of hidden layers     | 3          |
| Hidden layer size           | 64         |
| Learning rate of actor      | 1e-4       |
| Learning rate of critic     | 1e-3       |
| Discount factor             | 0.95       |
| Soft-update rate            | 0.01       |
| Initial exploration rate    | 0.1 (none in code)       |
| Initial noise rate          | 0.1        |
| Replay buffer size          | 5e5        |
| Batch size                  | 256        |

### Environment,Observation and Reward
Environment:The code leverages **Gymnasium**, **PettingZoo**, and **Pygame** as the primary tools for building and managing the environment.


Observation:Each agent can observe a maximum of **6 allies** and **6 adversaries** within its perception range. If the number of agents exceeds this limit, **the farthest ones are excluded from the observation**. Conversely, if the number of agents is below the threshold, the remaining portion of the observation vector is **padded with zeros**. This ensures a consistent observation structure regardless of the number of agents in range.The observation includes **the agent's own position, velocity, and heading, as well as the relative positions and headings of observed agents**.

Reward:
- **Prey Rewards**:  
  - The prey receives a reward **\( r = -1 \)** if caught by a predator.  
  - When caught by predator,the prey will not be eliminated but continuously minor it's reward as a situation of bleeding.A survival reward is given but returns to zero upon separation from predators.
  - Movement incurs a penalty: **\( -0.01|a<sub>F</sub>| - 0.1|a<sub>R</sub>| \)**, discouraging unnecessary motion.

- **Predator Rewards**:  
  - The predator receives a reward **\( r = +1 \)** for catching prey.  
  - Prey agents are not removed from the simulation after being caught, allowing continuous interaction.

- **Boundary Penalty**:  
  - A penalty of **\( -0.1 \)** is applied when agents contact boundaries, simulating environmental risks or confined spaces.


### Simulation and Evaluation

![image](https://github.com/xxnnnnn/PredatorPrey_RL_Reproduction/blob/main/eval_videos/gif%201_1.gif))
![image](https://github.com/xxnnnnn/PredatorPrey_RL_Reproduction/blob/main/eval_videos/gif%201_2.gif))
![image](https://github.com/xxnnnnn/PredatorPrey_RL_Reproduction/blob/main/eval_videos/gif%201_1.gif))


## Installation
git clone https://github.com/xxnnnnn/PredatorPrey_RL_Reproduction.git

cd PredatorPrey_RL_Reproduction

pip install -r requirements.txt
