# blokus_env/renderer.py

import pygame
import numpy as np
from blokus_env.constants import BOARD_SIZE, PLAYER_COLORS
from IPython import embed

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption('Blokus')
        self.clock = pygame.time.Clock()

    def render(self, board, pieces):
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.draw_pieces(board)
        pygame.display.flip()
        self.clock.tick(30)
        return np.array(pygame.surfarray.array3d(self.screen))

    def draw_grid(self):
        for x in range(0, 600, 30):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, 600))
        for y in range(0, 600, 30):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (600, y))

    def draw_pieces(self, board):
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x, y] != 0:
                    color = PLAYER_COLORS[board[x, y] - 1]
                    pygame.draw.rect(
                        self.screen, color,
                        pygame.Rect(y * 30, x * 30, 30, 30)
                    )
