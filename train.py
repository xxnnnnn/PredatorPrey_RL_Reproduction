import torch
import imageio  # Used for saving videos
import os  # Used for managing file paths
from predator_prey_env import Env  # Custom environment (predator-prey setup)
from maddpg import MADDPG  # Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
import numpy as np


def train():
    """
    Main training function for the MADDPG algorithm in a multi-agent predator-prey environment.
    It trains two sets of agents (predators and preys) simultaneously, saving model checkpoints and video
    recordings of the training process periodically.
    """

    # Initialize the device for computation (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the environment
    env = Env()
    num_episodes = 5000  # Number of episodes to train
    max_steps = env.max_steps  # Maximum number of steps per episode

    # Create directory to store video files
    video_dir = "./videos"
    os.makedirs(video_dir, exist_ok=True)

    # Retrieve environment details
    state_dim = env.observation_spaces[env.agents[0]].shape[0]  # State dimension for an agent
    action_dim = env.action_spaces[env.agents[0]].shape[0]  # Action dimension for an agent


    # Initialize the MADDPG algorithm
    maddpg = MADDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,  # Size of hidden layers in the neural network
        actor_lr=1e-4,  # Learning rate for actor networks
        critic_lr=1e-3,  # Learning rate for critic networks
        gamma=0.95,  # Discount factor for future rewards
        tau=0.01,  # Soft update parameter for target networks
        buffer_size=500000,  # Size of the experience replay buffer
        batch_size=256,  # Batch size for training
        device=device
    )

    # Attempt to load pre-trained models if available
    try:
        maddpg.predator_actor.load_state_dict(
            torch.load("predator_actor_500.pth", map_location=device, weights_only=True))
        maddpg.prey_actor.load_state_dict(torch.load("prey_actor_500.pth", map_location=device, weights_only=True))
        maddpg.predator_critic.load_state_dict(
            torch.load("predator_critic_500.pth", map_location=device, weights_only=True))
        maddpg.prey_critic.load_state_dict(torch.load("prey_critic_500.pth", map_location=device, weights_only=True))
        print("Loaded pre-trained models.")
    except FileNotFoundError:
        print("Pre-trained models not found. Starting training from scratch.")
    except TypeError as e:
        print(f"Error loading models: {e}")
        print("Ensure you are using a compatible version of PyTorch that supports 'weights_only=True'.")

    # Main training loop
    for episode in range(num_episodes):
        # Reset the environment for a new episode
        states, infos = env.reset()
        total_rewards = {'predator': 0, 'prey': 0}  # Initialize total rewards for both agents

        # Initialize a list to store frames for video recording
        frames = []

        for step in range(max_steps):
            # Select actions for predators based on current states
            predator_states = {agent: states[agent] for agent in env.predators}
            predator_actions = maddpg.select_action(predator_states, agent_type='predator', add_noise=True)

            # Select actions for preys based on current states
            prey_states = {agent: states[agent] for agent in env.preys}
            prey_actions = maddpg.select_action(prey_states, agent_type='prey', add_noise=True)

            # Combine actions from both predator and prey agents
            actions = {**predator_actions, **prey_actions}

            # Step the environment with the chosen actions
            next_states, rewards, dones, truncated, infos = env.step(actions)

            # Store the experiences in the replay buffer (separate for predator and prey agents)
            maddpg.replay_buffer.add(
                agent_type='predator',
                states=predator_states,
                actions=predator_actions,
                rewards={agent: rewards[agent] for agent in env.predators},
                next_states={agent: next_states[agent] for agent in env.predators},
                dones={agent: dones[agent] for agent in env.predators}
            )

            maddpg.replay_buffer.add(
                agent_type='prey',
                states=prey_states,
                actions=prey_actions,
                rewards={agent: rewards[agent] for agent in env.preys},
                next_states={agent: next_states[agent] for agent in env.preys},
                dones={agent: dones[agent] for agent in env.preys}
            )

            # Train the agents after collecting enough experiences in the replay buffer
            if len(maddpg.replay_buffer.predator_buffer) > maddpg.replay_buffer.predator_buffer.batch_size:
                maddpg.train_step(agent_type='predator')
                maddpg.train_step(agent_type='prey')

            # Update the state to the next state
            states = next_states

            # Accumulate the rewards for each agent
            total_rewards['predator'] += sum(rewards[predator] for predator in env.predators)
            total_rewards['prey'] += sum(rewards[prey] for prey in env.preys)

            # Check if all agents are done (i.e., episode is over)
            if all(dones.values()):
                break

            # Render the environment and capture the frame for video
            frame = env.render()  # Ensure render method returns RGB image
            if (episode+1) % 100 == 0:
                # Save models every 500 episodes
                if (episode + 1) % 500 == 0:
                    torch.save(maddpg.predator_actor.state_dict(), f"predator_actor_{episode + 1}.pth")
                    torch.save(maddpg.prey_actor.state_dict(), f"prey_actor_{episode + 1}.pth")
                    torch.save(maddpg.predator_critic.state_dict(), f"predator_critic_{episode + 1}.pth")
                    torch.save(maddpg.prey_critic.state_dict(), f"prey_critic_{episode + 1}.pth")

                # Ensure the frame's dimensions are multiples of 16 for video encoding
                if frame is not None:
                    current_height, current_width = frame.shape[:2]
                    macro_block_size = 16

                    # Calculate the target height and width
                    desired_height = int(np.ceil(current_height / macro_block_size)) * macro_block_size
                    desired_width = int(np.ceil(current_width / macro_block_size)) * macro_block_size

                    # Calculate the padding needed
                    padding_height_needed = desired_height - current_height
                    padding_width_needed = desired_width - current_width

                    # Apply padding to the frame (top, bottom, left, right)
                    pad_top = padding_height_needed // 2
                    pad_bottom = padding_height_needed - pad_top
                    pad_left = padding_width_needed // 2
                    pad_right = padding_width_needed - pad_left

                    # Pad the frame with black pixels (constant mode)
                    frame = np.pad(
                        frame,
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),  # Padding in height, width, and channel
                        mode='constant',
                        constant_values=0  # Black padding
                    )

                    # Add the padded frame to the list of frames
                    frames.append(frame)

        # Save video every 100 episodes
        if frames and ((episode+1) % 100 == 0):
            video_path = os.path.join(video_dir, f"episode_{episode}.mp4")
            imageio.mimsave(video_path, frames, fps=100, codec="libx264")  # Adjust fps as needed
            print(f"Saved video for Episode {episode} at {video_path}")
            frames = []  # Clear frames list after saving

        env.epoch_counter += 1  # Increment the epoch counter

        # Log the total rewards for predators and preys in the current episode
        print(
            f"Episode {episode+1}/{num_episodes} - Total Predator Reward: {total_rewards['predator']}, Total Prey Reward: {total_rewards['prey']}")

    # Optionally: Save final trained models after training
    torch.save(maddpg.predator_actor.state_dict(), "predator_actor.pth")
    torch.save(maddpg.predator_critic.state_dict(), "predator_critic.pth")
    torch.save(maddpg.prey_actor.state_dict(), "prey_actor.pth")
    torch.save(maddpg.prey_critic.state_dict(), "prey_critic.pth")

    env.close()  # Close the environment at the end of training


if __name__ == "__main__":
    train()
