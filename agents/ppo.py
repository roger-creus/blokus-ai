import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces

# Import the Blokus environment
from blokus_env.blokus_env import BlokusEnv
from blokus_env.constants import BOARD_SIZE, NUM_PLAYERS, INITIAL_POSITIONS, PLAYER_COLORS, PIECES
from IPython import embed


# Important: Supress the pygame output
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

def record_video_every_fn(episode_id: int) -> bool:
    return episode_id % 3 == 0

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Blokus-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

# Save the model
def save_model(agent, path):
    torch.save(agent.state_dict(), path)

# Load the model
def load_model(agent, path):
    agent.load_state_dict(torch.load(path))

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = BlokusEnv(render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=record_video_every_fn)
        return env

    return thunk

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Convolutional network for the board state
        self.board_network = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected network for the pieces
        self.pieces_network = nn.Sequential(
            nn.Linear(len(PIECES) * NUM_PLAYERS, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.current_player_network = nn.Sequential(
            nn.Linear(NUM_PLAYERS, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # Combined network
        self.fc1 = nn.Linear(16 + 128 + 64 * 20 * 20, 512)
        self.relu = nn.ReLU()

        # Define the action dimensions
        self.action_dims = [
            envs.single_action_space.spaces[i].n for i in range(len(envs.single_action_space.spaces))
        ]
        self.actors = nn.ModuleList([
            nn.Linear(512, action_dim) for action_dim in self.action_dims
        ])
        self.critic = nn.Linear(512, 1)

    def forward(self, board, pieces, current_player):
        # Process the board and pieces separately
        board_features = self.board_network(board)
        pieces_features = self.pieces_network(pieces)
        current_player_features = self.current_player_network(current_player)

        # Combine the features
        combined = torch.cat((board_features, pieces_features, current_player_features), dim=-1)
        hidden = self.relu(self.fc1(combined))

        return hidden

    def get_value(self, board, pieces, current_player):
        hidden = self.forward(board, pieces, current_player)
        return self.critic(hidden)

    def get_action_and_value(self, board, pieces, current_player, action=None):
        hidden = self.forward(board, pieces,current_player)
        logits = [actor(hidden) for actor in self.actors]
        probs = [Categorical(logits=logit) for logit in logits]

        if action is None:
            action = torch.stack([prob.sample() for prob in probs], dim=-1)

        logprobs = torch.stack([prob.log_prob(act) for prob, act in zip(probs, action.T)], dim=-1).sum(dim=-1)
        entropy = torch.stack([prob.entropy() for prob in probs], dim=-1).sum(dim=-1)

        return action, logprobs, entropy, self.critic(hidden)
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, spaces.Tuple), "only Tuple action space is supported"

    # Self-Play
    old_agents = []
    save_path = f"models/{run_name}_latest.pt"
    old_model_save_path = f"models/{run_name}_old_{len(old_agents)}.pt"

    agent = Agent(envs).to(device)
    
    print("Agent Architecture")
    print(agent)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = {
        'board': torch.zeros((args.num_steps, args.num_envs, 1, BOARD_SIZE, BOARD_SIZE)).to(device),
        'pieces': torch.zeros((args.num_steps, args.num_envs, len(PIECES) * NUM_PLAYERS)).to(device),
        'current_player': torch.zeros((args.num_steps, args.num_envs, NUM_PLAYERS)).to(device)
    }
    actions = torch.zeros((args.num_steps, args.num_envs, len(envs.single_action_space.spaces))).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs_board = torch.Tensor(next_obs['board']).unsqueeze(1).to(device)
    next_obs_pieces = torch.Tensor(next_obs['pieces']).to(device)
    next_obs_current_player = torch.Tensor(next_obs['current_player']).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs['board'][step] = next_obs_board
            obs['pieces'][step] = next_obs_pieces
            obs["current_player"][step] = next_obs_current_player
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_board, next_obs_pieces, next_obs_current_player)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy().T)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs_board = torch.Tensor(next_obs['board']).unsqueeze(1).to(device)
            next_obs_pieces = torch.Tensor(next_obs['pieces']).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_board, next_obs_pieces, next_obs_current_player).reshape(-1, 1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs_board = obs['board'].reshape((-1, 1, BOARD_SIZE, BOARD_SIZE))
        b_obs_pieces = obs['pieces'].reshape((-1, len(PIECES) * NUM_PLAYERS))
        b_obs_current_player = obs['current_player'].reshape((-1, NUM_PLAYERS))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, len(envs.single_action_space.spaces)))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprobs, entropy, newvalues = agent.get_action_and_value(
                    b_obs_board[mb_inds], b_obs_pieces[mb_inds], b_obs_current_player[mb_inds], b_actions[mb_inds]
                )
                newvalues = newvalues.view(-1)
                logratio = newlogprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate the approx_kl
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = (newvalues - b_returns[mb_inds]).pow(2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("charts/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/explained_variance", explained_var, global_step)
        
        print(f"PPO Iteration {iteration} completed. Total steps: {global_step}")

    envs.close()
    writer.close()