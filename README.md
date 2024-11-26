# PredatorPrey_RL_Reproduction

This project is a reproduction of the research conducted by Shiyu Zhao, a researcher at Westlake University. Below are the details of the original article:

**Li, J., Li, L., & Zhao, S.** (2023). "Predator–prey survival pressure is sufficient to evolve swarming behaviors." *New Journal of Physics*, 25(9), 092001. 

The article can be accessed here: https://iopscience.iop.org/article/10.1088/1367-2630/acf33a

Additionally,there are also some videos that discuss concepts similar to those presented in the article.The video links are listed below.

- [Evolving AIs - Predator vs Prey, who will win?](https://www.youtube.com/watch?v=qwrp3lB-jkQ&t=592s)
- [Much bigger simulation, AIs learn Phalanx](https://www.youtube.com/watch?v=tVNoetVLuQg&t=815s)

## Article overview
The main idea of this article is that when using reinforcement learning (RL) to simulate swarm behavior, specific reward functions may cause the swarm to exhibit certain behaviors. However, such tailored rewards are not necessary. Instead, the coevolution of the swarm can be achieved using the simplest form of reward—the pressure of survival.

## Code Description
project_directory/

├── gui.py                  # Code for the graphical user interface.

├── maddpg.py               # Implementation of the MADDPG algorithm used in the project.

├── main.py                 # Main script to run the program.

├── networks.py             # Defines the actor-critic network, shared among agents of the same type.

├── predator_prey_env.py    # Implementation of the predator-prey simulation environment.

├── replay_buffer.py        # Replay buffer shared by agents of the same type for experience replay.

It is worth noting that there are some simplifications and omissions in my code. However, these modifications are unrelated to the core content of the article.

For example, the article prohibits agents from colliding by using Hooke's law, but this aspect has been omitted in the code. Additionally, the output of the actor network has been scaled in the code to ensure smoother trajectories during simulation. 

These changes do not affect the fundamental essence of the article.

### Homogeneity
Since the game involves only two species—prey and predator—it is intuitive that agents within the same species share identical strategies, as they belong to the same ecological role. However, individual agents may perform distinct actions based on differences in their observations. To simplify the structure, agents of the same species utilize a shared Actor-Critic network and replay buffer, reducing redundancy while maintaining consistency in decision-making.

### Neural Networks and Replay Buffer
This article implements the Actor-Critic network, where the actor network selects the agent's action based on its current observation. The actor's outputs have two dimensions: \( a_f \) (angular acceleration) and \( a_R \) (radial acceleration). The critic network evaluates the expected future reward corresponding to the chosen action, enabling the agent to optimize its strategy over time.

The replay buffer plays a crucial role in enabling the shared learning architecture. For each conspecies, a single replay buffer is maintained, ensuring uniform data collection and utilization across all agents. During training, experiences from all agents of the same species are aggregated into the shared buffer. This collective learning framework allows agents to benefit from the experiences of their peers, leading to more robust and generalized strategies.

The network consists of three hidden layers, each with 64 units. The **Critic Network** outputs a single value representing the expected reward, while the **Actor Network** outputs two values: angular acceleration (\(a_F\)) and radial acceleration (\(a_R\)).
 

### The Hyper-parameter
| **Hyper-parameter**         | **Value**  |
|-----------------------------|------------|
| Number of episodes          | 2000       |
| Episode length              | 100        |
| Number of hidden layers     | 3          |
| Hidden layer size           | 64         |
| Learning rate of actor      | \(1 \times 10^{-4}\) |
| Learning rate of critic     | \(1 \times 10^{-3}\) |
| Discount factor             | 0.95       |
| Soft-update rate            | 0.01       |
| Initial exploration rate    | 0.1        |
| Initial noise rate          | 0.1        |
| Replay buffer size          | \(5 \times 10^5\) |
| Batch size                  | 256        |

### Observation and Reward



### Simulation and Evaluation




## Installation
git clone https://github.com/xxnnnnn/PredatorPrey_RL_Reproduction.git

cd PredatorPrey_RL_Reproduction

pip install -r requirements.txt
