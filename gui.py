# gui.py
import pygame
from typing import Dict
import numpy as np


class Gui:
    def __init__(self, width: int, height: int, grid_spacing_light: int = 50, grid_spacing_dark: int = 100):
        """
        Initialize the GUI with environment dimensions and grid settings.
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Predator-Prey Environment")

        # Grid settings
        self.grid_spacing_light = grid_spacing_light
        self.grid_spacing_dark = grid_spacing_dark
        self.light_grid_color = (200, 200, 200)  # Light gray
        self.dark_grid_color = (150, 150, 150)  # Darker gray

        # Font settings
        self.font = pygame.font.Font(None, 20)

        # Colors for agents
        self.predator_color = (255, 0, 0)  # Red
        self.prey_color = (0, 255, 0)  # Green

        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        self.fps = 100  # Frames per second

    def draw_grid(self):
        """
        Draws a grid on the screen with light and dark lines.
        """
        # Draw light grid lines
        for x in range(0, self.width, self.grid_spacing_light):
            pygame.draw.line(self.screen, self.light_grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, self.grid_spacing_light):
            pygame.draw.line(self.screen, self.light_grid_color, (0, y), (self.width, y))

        # Draw dark grid lines
        for x in range(0, self.width, self.grid_spacing_dark):
            pygame.draw.line(self.screen, self.dark_grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, self.grid_spacing_dark):
            pygame.draw.line(self.screen, self.dark_grid_color, (0, y), (self.width, y))

    def draw_agents(self, agent_positions: Dict[str, np.ndarray], agent_headings: Dict[str, float], agent_radius: Dict[str, float]):
        """
        Draws predators and preys on the screen with their headings.

        :param agent_positions: Dictionary mapping agent IDs to their positions.
        :param agent_headings: Dictionary mapping agent IDs to their headings in radians.
        """
        for agent_id, position in agent_positions.items():
            x, y = position
            heading = agent_headings.get(agent_id, 0)  # Get agent's heading, default to 0
            if 'predator' in agent_id:
                color = self.predator_color
            else:
                color = self.prey_color

            # Draw agent as a circle
            radius = int(agent_radius.get(agent_id, 10))
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)

            # Draw heading as a line (nose)
            line_length = 12  # Length of the heading line
            end_x = int(x + line_length * np.cos(heading))
            end_y = int(y + line_length * np.sin(heading))
            pygame.draw.line(self.screen, (0, 0, 0), (int(x), int(y)), (end_x, end_y), 2)

    def update_display(self, agent_positions: Dict[str, np.ndarray], agent_headings: Dict[str, float], epoch: int,agent_radius:Dict[str,float],total_epoch = 5000,eval=False):
        """
        Updates the entire display with grid and agents.

        :param agent_positions: Dictionary mapping agent IDs to their positions.
        :param agent_headings: Dictionary mapping agent IDs to their headings.
        :param epoch: Current epoch number.
        """

        self.screen.fill((255, 255, 255))  # Fill the screen with white
        self.draw_grid()
        self.draw_agents(agent_positions, agent_headings,agent_radius)

        # Draw epoch information in the top-left corner
        if eval:
            epoch_text = self.font.render(f"Eval", True, (0, 0, 0))
        else:
            epoch_text = self.font.render(f"Epoch: {epoch}/{total_epoch}", True, (0, 0, 0))
        self.screen.blit(epoch_text, (6, 6))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def handle_events(self):
        """
        Handles Pygame events to allow window closing.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def close(self):
        """
        Closes the Pygame window.
        """
        pygame.quit()
