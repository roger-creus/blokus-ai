# Blokus-AI: A Gymnasium Environment for the Blokus Board Game

Blokus-AI is an implementation of the Blokus board game environment using the Gymnasium framework. This environment is designed to be used for training AI agents to play Blokus.

<div style="display:flex;">
  <img src="demo_0.gif" style="flex:1;margin-right:5px;width:45%;" />
  <img src="demo_1.gif" style="flex:1;margin-left:5px;width:45%;" />
</div>

# 🚀 Usage

```bash
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
```

# 🛠️ Installation
To set up the environment, follow these steps:

```bash
conda create -n blokus python=3.8
conda activate blokus
pip install -r requirements.txt
pip install -e .
```

# 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

# 📚 Citation

If you use Blokus-AI in your research, please cite it as follows:

```bash
@misc{blokus-ai,
    author = {Roger Creus Castanyer},
    title = {Blokus-AI: A Gymnasium Environment for the Blokus Board Game},
    year = {2024},
}
```