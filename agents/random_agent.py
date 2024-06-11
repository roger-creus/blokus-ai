# agents/random_agent.py

import random
import numpy as np

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        valid_actions = self.env.game_logic.get_valid_actions(self.env.board, self.env.pieces, self.env.current_player)
        if valid_actions:
            return random.choice(valid_actions)
        return None
