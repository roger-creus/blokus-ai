# blokus_env/blokus_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from blokus_env.constants import BOARD_SIZE, NUM_PLAYERS, INITIAL_POSITIONS, PLAYER_COLORS, PIECES
from blokus_env.game_logic import GameLogic
from blokus_env.renderer import Renderer
from IPython import embed
import matplotlib.pyplot as plt

class BlokusEnv(gym.Env):
    def __init__(self, render_mode='rgb_array'):
        self.metadata = {'render.modes': ['human', 'rgb_array'], "render_fps": 1}
        self.render_mode = render_mode
        super(BlokusEnv, self).__init__()

        # Define the observation space: the board state and the pieces each player has
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=NUM_PLAYERS, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
            'pieces': spaces.MultiBinary(len(PIECES) * NUM_PLAYERS),
            'current_player': spaces.MultiBinary(NUM_PLAYERS)
        })

        # Define the action space: (piece index, x, y, rotation, horizontal_flip, vertical_flip)
        self.action_space = spaces.Tuple((
            spaces.Discrete(len(PIECES)),  # Piece index or None
            spaces.Discrete(BOARD_SIZE),
            spaces.Discrete(BOARD_SIZE),
            spaces.Discrete(4),  # 4 rotations (0, 1, 2, 3 degrees)
            spaces.Discrete(2),  # Horizontal flip (0 or 1)
            spaces.Discrete(2)   # Vertical flip (0 or 1)
        ))

        self.game_logic = GameLogic()
        self.renderer = Renderer()
        self.reset()

    def reset(self, seed=None, options=None):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.pieces = np.ones((NUM_PLAYERS, len(PIECES)), dtype=np.int8)
        self.current_player = 1
        return self._get_observation(), {}

    def step(self, action):
        all_valid_actions = self.game_logic.get_valid_actions(self.board, self.pieces, self.current_player)
        
        if len(all_valid_actions) == 0:
            reward = -10
        else:
            piece_index, x, y, rotation, horizontal_flip, vertical_flip = action
            
            valid_move = self.game_logic.place_piece(
                self.board, self.pieces, self.current_player, piece_index, x, y, rotation,
                horizontal_flip=horizontal_flip, vertical_flip=vertical_flip
            )

            if not valid_move:
                action = all_valid_actions[np.random.randint(len(all_valid_actions))]
                piece_index, x, y, rotation, horizontal_flip, vertical_flip = action
                
                valid_move = self.game_logic.place_piece(
                    self.board, self.pieces, self.current_player, piece_index, x, y, rotation,
                    horizontal_flip=horizontal_flip, vertical_flip=vertical_flip
                )                
                reward = -5  # Invalid move penalty
            else:
                piece_name = list(PIECES.keys())[piece_index]
                reward = len(PIECES[piece_name])
            
        self.current_player = (self.current_player % NUM_PLAYERS) + 1

        done = self.game_logic.is_game_over(self.board, self.pieces)
        observation = self._get_observation()
        info = {}
        return observation, reward, done, done, info

    def render(self, mode='human'):
        img = self.renderer.render(self.board, self.pieces)
        return img

    def _get_observation(self):
        return {
            'board': self.board,
            'pieces': self.pieces.flatten().astype(np.int8),
            'current_player': np.eye(NUM_PLAYERS)[self.current_player - 1].flatten().astype(np.int8)
        }
    
