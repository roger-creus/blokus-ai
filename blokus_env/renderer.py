import pygame
import numpy as np
from blokus_env.constants import PLAYER_COLORS

class Renderer:
    def __init__(self, options):
        """
        Initialize the Renderer with a Pygame screen and set up the display.
        """
        pygame.init()
        self.board_size = options["board_size"]

        self.window_size = self.board_size * 30

        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Blokus')
        self.clock = pygame.time.Clock()

    def render(self, board, pieces):
        """
        Render the current state of the board.

        Parameters:
        - board (np.ndarray): The current board state.
        - pieces (np.ndarray): The pieces owned by each player (not used in this function).

        Returns:
        - np.ndarray: The rendered image as a 3D numpy array.
        """
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.draw_pieces(board)
        pygame.display.flip()
        self.clock.tick(30)
        return np.array(pygame.surfarray.array3d(self.screen))

    def draw_grid(self):
        """
        Draw the grid lines on the board.
        """
        for x in range(0, self.window_size, 30):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, 30):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.window_size, y))

    def draw_pieces(self, board):
        """
        Draw the pieces on the board according to the board state.

        Parameters:
        - board (np.ndarray): The current board state.
        """
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x, y] != 0:
                    color = PLAYER_COLORS[board[x, y] - 1]
                    pygame.draw.rect(
                        self.screen, color,
                        pygame.Rect(y * 30, x * 30, 30, 30)
                    )
