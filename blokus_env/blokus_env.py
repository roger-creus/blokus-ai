import gymnasium as gym
import numpy as np
from gymnasium import spaces
from blokus_env.constants import BOARD_SIZE, NUM_PLAYERS, PIECES
from blokus_env.game_logic import GameLogic
from blokus_env.renderer import Renderer

class BlokusEnv(gym.Env):
    """
    Implementation of the Blokus board game as a Gymnasium RL environment.
    """
    def __init__(self, render_mode='rgb_array'):
        """
        Initialize the Blokus environment.

        Parameters:
        - render_mode (str): Mode to render with ('human' or 'rgb_array').
        """
        self.metadata = {'render.modes': ['human', 'rgb_array'], "render_fps": 10}
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
            spaces.Discrete(len(PIECES)),  # Piece index
            spaces.Discrete(BOARD_SIZE),   # X position
            spaces.Discrete(BOARD_SIZE),   # Y position
            spaces.Discrete(4),            # 4 rotations (0, 1, 2, 3 degrees)
            spaces.Discrete(2),            # Horizontal flip (0 or 1)
            spaces.Discrete(2)             # Vertical flip (0 or 1)
        ))

        self.game_logic = GameLogic()
        self.renderer = Renderer()
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and return an initial observation.

        Parameters:
        - seed (int, optional): Seed for random number generator.
        - options (dict, optional): Additional options for reset.

        Returns:
        - observation (dict): Initial observation of the environment.
        - info (dict): Additional info.
        """
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.pieces = np.ones((NUM_PLAYERS, len(PIECES)), dtype=np.int8)
        self.current_player = 1
        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one step in the environment.

        Parameters:
        - action (tuple): Action to be taken in the environment.

        Returns:
        - observation (dict): The observation after taking the action.
        - reward (float): Reward received after taking the action.
        - done (bool): Whether the episode has ended.
        - info (dict): Additional info.
        """
        # Compute 1 valid action for the current player
        all_valid_actions = self.game_logic.get_valid_actions(self.board, self.pieces, self.current_player)
        
        # If the player has no valid actions, negative reward
        if len(all_valid_actions) == 0:
            reward = -10
        else:
            piece_index, x, y, rotation, horizontal_flip, vertical_flip = action
            
            # Make sure the action is valid
            valid_move = self.game_logic.place_piece(
                self.board, self.pieces, self.current_player, piece_index, x, y, rotation,
                horizontal_flip=horizontal_flip, vertical_flip=vertical_flip
            )

            # If not valid, choose a random valid action and set negative reward
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
        """
        Render the environment.

        Parameters:
        - mode (str): The mode to render with. ('human' or 'rgb_array').

        Returns:
        - img (np.ndarray): Rendered image if mode is 'rgb_array'.
        """
        img = self.renderer.render(self.board, self.pieces)
        return img

    def _get_observation(self):
        """
        Get the current observation of the environment.

        Returns:
        - observation (dict): The current observation of the environment.
        """
        return {
            'board': self.board,
            'pieces': self.pieces.flatten().astype(np.int8),
            'current_player': np.eye(NUM_PLAYERS)[self.current_player - 1].flatten().astype(np.int8)
        }