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

It is worth noting that there are some simplifications and omissions in my code. However, these modifications are unrelated to the core content of the article. For example, the article prohibits agents from colliding by using Hooke's law, but this aspect has been omitted in the code. Additionally, the output of the actor network has been scaled in the code to ensure smoother trajectories during simulation. These changes do not affect the fundamental essence of the article.

## Installation
git clone https://github.com/xxnnnnn/PredatorPrey_RL_Reproduction.git
cd PredatorPrey_RL_Reproduction
pip install -r requirements.txt
