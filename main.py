# main.py

import gymnasium as gym
import blokus_env
from agents.random_agent import RandomAgent

def main():
    env = gym.make('blokus_env:BlokusEnv-v0')

    # Create a random agent for each player
    agents = [RandomAgent(env) for _ in range(4)]

    obs = env.reset()
    done = False

    print("Starting game")
    while not done:
        # Get the current player from the environment
        current_player = env.current_player - 1  # Player index in agents list
        action = agents[current_player].get_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()

    print("Game Over")
    env.close()

if __name__ == '__main__':
    main()
