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
    def __init__(self, render_mode='rgb_array', options=None):
        """
        Initialize the Blokus environment.

        Parameters:
        - render_mode (str): Mode to render with ('human' or 'rgb_array').
        """
        self.metadata = {"render_modes": ['human', 'rgb_array'], "render_fps": 10}
        # assert render_mode in self.metadata['render_modes'], f"Invalid render mode: {render_mode}"
        self.render_mode = render_mode
        # Manage the options for the environment (in case of custom board size, number of players, etc...)
        if options is None:
            self.options = {
                "board_size": BOARD_SIZE,
                "players": NUM_PLAYERS
            }
        else:
            self.options = options
            self.options.setdefault("board_size", BOARD_SIZE)
            self.options.setdefault("players", NUM_PLAYERS)
            if self.options["players"] > NUM_PLAYERS or self.options["players"] < 1:
                raise ValueError(f"Invalid number of players: {self.options['players']}. Must be between 1 and {NUM_PLAYERS}.")



        super(BlokusEnv, self).__init__()

        print("BlokusEnv initialized with options:", self.options)
        # Define the observation space: the board state and the pieces each player has
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=self.options["players"], shape=(self.options["board_size"], self.options["board_size"]), dtype=np.int8),
            'pieces': spaces.MultiBinary(len(PIECES) * self.options["players"]),
            'current_player': spaces.MultiBinary(self.options["players"])
        })

        # Define the action space: (piece index, x, y, rotation, horizontal_flip, vertical_flip)
        self.action_space = spaces.Tuple((
            spaces.Discrete(len(PIECES)),  # Piece index
            spaces.Discrete(self.options["board_size"]),   # X position
            spaces.Discrete(self.options["board_size"]),   # Y position
            spaces.Discrete(4),            # 4 rotations (0, 1, 2, 3 degrees)
            spaces.Discrete(2),            # Horizontal flip (0 or 1)
            spaces.Discrete(2)             # Vertical flip (0 or 1)
        ))

        # Initialize the scores for each player
        self.scores = np.zeros(self.options["players"], dtype=np.int8)

        # Initialize the game logic and renderer
        self.game_logic = GameLogic(options=self.options)
        self.renderer = Renderer(options=self.options)
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
        self.board = np.zeros((self.options["board_size"], self.options["board_size"]), dtype=np.int8)
        self.pieces = np.ones((self.options["players"], len(PIECES)), dtype=np.int8)
        self.current_player = 1
        return self._get_observation(), {"scores": self.scores}

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
                self.scores[self.current_player - 1] += len(PIECES[list(PIECES.keys())[piece_index]])
            else:
                piece_name = list(PIECES.keys())[piece_index]
                reward = len(PIECES[piece_name])

                self.scores[self.current_player - 1] += reward*2
            
        self.current_player = (self.current_player % self.options["players"]) + 1
        done = self.game_logic.is_game_over(self.board, self.pieces)
        observation = self._get_observation()
        info = {'scores': self.scores}
        return observation, reward, done, done, info

    def render(self):
        """
        Render the environment.

        Parameters:
        - mode (str): The mode to render with. ('human' or 'rgb_array').

        Returns:
        - img (np.ndarray): Rendered image if mode is 'rgb_array'.
        """
        self.renderer.render(self.board, self.pieces)


    def _get_observation(self):
        """
        Get the current observation of the environment.

        Returns:
        - observation (dict): The current observation of the environment.
        """
        return {
            'board': self.board,
            'pieces': self.pieces.flatten().astype(np.int8),
            'current_player': np.eye(self.options["players"])[self.current_player - 1].flatten().astype(np.int8)
        }