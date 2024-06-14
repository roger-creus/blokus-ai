import blokus_env
import gymnasium as gym

# Create the environment
env = gym.make("BlokusEnv-v0", render_mode="human")

# Reset the environment
obs, _ = env.reset()
done = False

while not done:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    done = term or trunc
    # Render the environment
    env.render()
