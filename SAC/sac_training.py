import json
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

import argparse

from env import Box, get_last_states
from model import CirclePF, CirclePB, NeuralNet
from sampling import (
    sample_trajectories,
    evaluate_backward_logprobs,
)
from sac import SAC
from sac_replay_memory import ReplayMemory, trajectories_to_transitions

from utils import (
    fit_kde,
    plot_reward,
    sample_from_reward,
    plot_samples,
    estimate_jsd,
    plot_trajectories,
    plot_termination_probabilities,
)

import sac_config as config

try:
    import wandb
except ModuleNotFoundError:
    pass


USE_WANDB = True
NO_PLOT = False

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default=config.DEVICE)
parser.add_argument("--dim", type=int, default=config.DIM)
parser.add_argument("--delta", type=float, default=config.DELTA)
parser.add_argument("--env_epsilon", type=float, default=config.ENV_EPSILON)
parser.add_argument(
    "--n_components",
    type=int,
    default=config.N_COMPONENTS,
    help="Number of components in Mixture Of Betas",
)

parser.add_argument("--reward_debug", action="store_true", default=config.REWARD_DEBUG)
parser.add_argument(
    "--reward_type",
    type=str,
    choices=["baseline", "ring", "angular_ring", "multi_ring", "curve", "gaussian_mixture"],
    default=config.REWARD_TYPE,
    help="Type of reward function to use. To modify reward-specific parameters (radius, sigma, etc.), edit rewards.py"
)
parser.add_argument(
    "--n_components_s0",
    type=int,
    default=config.N_COMPONENTS_S0,
    help="Number of components in Mixture Of Betas",
)
parser.add_argument(
    "--beta_min",
    type=float,
    default=config.BETA_MIN,
    help="Minimum value for the concentration parameters of the Beta distribution",
)
parser.add_argument(
    "--beta_max",
    type=float,
    default=config.BETA_MAX,
    help="Maximum value for the concentration parameters of the Beta distribution",
)
parser.add_argument(
    "--PB",
    type=str,
    choices=["learnable", "tied", "uniform"],
    default=config.PB,
    help="Backward policy type",
)
parser.add_argument("--gamma_scheduler", type=float, default=config.GAMMA_SCHEDULER)
parser.add_argument("--scheduler_milestone", type=int, default=config.SCHEDULER_MILESTONE)
parser.add_argument("--seed", type=int, default=config.SEED)
parser.add_argument("--lr", type=float, default=config.LR, help="Learning rate for SAC")
parser.add_argument("--BS", type=int, default=config.BS)
parser.add_argument("--n_iterations", type=int, default=config.N_ITERATIONS)
parser.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM)
parser.add_argument("--n_hidden", type=int, default=config.N_HIDDEN)
parser.add_argument("--n_evaluation_trajectories", type=int, default=config.N_EVALUATION_TRAJECTORIES)
parser.add_argument("--no_plot", action="store_true", default=config.NO_PLOT)
parser.add_argument("--no_wandb", action="store_true", default=config.NO_WANDB)
parser.add_argument("--wandb_project", type=str, default=config.WANDB_PROJECT)

# SAC-specific arguments
parser.add_argument("--policy", type=str, default="Gaussian", help="Policy type")
parser.add_argument("--tau", type=float, default=config.TAU, help="Target smoothing coefficient")
parser.add_argument("--sac_alpha", type=float, default=config.SAC_ALPHA, help="SAC temperature parameter")
parser.add_argument("--automatic_entropy_tuning", action="store_true", default=config.AUTOMATIC_ENTROPY_TUNING, help="Automatically adjust alpha")
parser.add_argument("--target_update_interval", type=int, default=config.TARGET_UPDATE_INTERVAL, help="Target network update interval")
parser.add_argument("--Critic_hidden_size", type=int, default=config.CRITIC_HIDDEN_SIZE, help="Hidden size for SAC critic networks")
parser.add_argument("--replay_size", type=int, default=config.REPLAY_SIZE, help="Replay buffer size")
parser.add_argument("--sac_batch_size", type=int, default=config.SAC_BATCH_SIZE, help="SAC batch size")
parser.add_argument("--updates_per_step", type=int, default=config.UPDATES_PER_STEP, help="SAC updates per step")

args = parser.parse_args()

if args.no_plot:
    NO_PLOT = True

if args.no_wandb:
    USE_WANDB = False

if USE_WANDB:
    wandb.init(project=args.wandb_project, save_code=True)
    wandb.config.update(args)

device = args.device
dim = args.dim
delta = args.delta
seed = args.seed
lr = args.lr
n_iterations = args.n_iterations
BS = args.BS
n_components = args.n_components
n_components_s0 = args.n_components_s0

if seed == 0:
    seed = np.random.randint(int(1e6))

run_name = f"SAC_d{delta}_{args.reward_type}_PB{args.PB}_lr{lr}_sd{seed}"
run_name += f"_n{n_components}_n0{n_components_s0}"
run_name += f"_gamma{args.gamma_scheduler}_mile{args.scheduler_milestone}"
run_name += f"_sac_alpha{args.sac_alpha}_batch{args.sac_batch_size}"
print(run_name)
if USE_WANDB:
    wandb.run.name = run_name  # type: ignore

torch.manual_seed(seed)
np.random.seed(seed)

print(f"Using device: {device}")

env = Box(
    dim=dim,
    delta=delta,
    epsilon=args.env_epsilon,
    device_str=device,
    reward_type=args.reward_type,
    reward_debug=args.reward_debug,
)

# Get the true KDE
samples = sample_from_reward(env, n_samples=10000)
true_kde, fig1 = fit_kde(samples, plot=True)

if USE_WANDB:
    # log the reward figure
    fig2 = plot_reward(env)

    wandb.log(
        {
            "reward": wandb.Image(fig2),
            "reward_kde": wandb.Image(fig1),
        }
    )


# Create SAC agent (includes CirclePF as policy)
sac_agent = SAC(args, env)

# Create replay memory
memory = ReplayMemory(args.replay_size, seed, device=device)

bw_model = CirclePB(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    torso=sac_agent.policy.torso if args.PB == "tied" else None,
    uniform=args.PB == "uniform",
    n_components=n_components,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
).to(device)

jsd = float("inf")
sac_updates = 0  # Track SAC update steps

for i in trange(1, n_iterations + 1):

    trajectories, actionss, logprobs, all_logprobs = sample_trajectories(
        env,
        sac_agent.policy,
        BS,
    )
    last_states = get_last_states(env, trajectories)
    logrewards = env.reward(last_states).log()
    bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(
        env, bw_model, trajectories
    )

    # Convert trajectories to transitions and push to replay memory
    all_states, all_actions, all_rewards, all_next_states, all_dones = trajectories_to_transitions(
        trajectories, actionss, all_bw_logprobs, last_states, logrewards, env
    )
    memory.push_batch(all_states, all_actions, all_rewards, all_next_states, all_dones)

    if len(memory) > args.sac_batch_size:
        for _ in range(args.updates_per_step):
            qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha = sac_agent.update_parameters(memory, args.sac_batch_size, sac_updates)
            sac_updates += 1
        # Step the scheduler once per iteration (not per update)
        sac_agent.policy_scheduler.step()

    if any(
        [
            torch.isnan(list(sac_agent.policy.parameters())[i]).any()
            for i in range(len(list(sac_agent.policy.parameters())))
        ]
    ):
        raise ValueError("NaN in model parameters")

    if i % 100 == 0:
        log_dict = {
            "sac/critic_1_loss": qf1_loss,
            "sac/critic_2_loss": qf2_loss,
            "sac/policy_loss": policy_loss,
            "sac/alpha_loss": alpha_loss,
            "sac/alpha": alpha,
            "sac/updates": sac_updates,
            "sac/replay_size": len(memory),
            "states_visited": i * BS,
        }

        # Evaluate JSD every 500 iterations and add to the same log
        if i % 500 == 0:
            trajectories, _, _, _ = sample_trajectories(
                env, sac_agent.policy, args.n_evaluation_trajectories
            )
            last_states = get_last_states(env, trajectories)
            kde, fig4 = fit_kde(last_states, plot=True)
            jsd = estimate_jsd(kde, true_kde)

            log_dict["JSD"] = jsd

            if not NO_PLOT:
                colors = plt.cm.rainbow(np.linspace(0, 1, 10))
                fig1 = plot_samples(last_states[:2000].detach().cpu().numpy())
                fig2 = plot_trajectories(trajectories.detach().cpu().numpy()[:20])
                fig3 = plot_termination_probabilities(sac_agent.policy)

                log_dict["last_states"] = wandb.Image(fig1)
                log_dict["trajectories"] = wandb.Image(fig2)
                log_dict["termination_probs"] = wandb.Image(fig3)
                log_dict["kde"] = wandb.Image(fig4)

        if USE_WANDB:
            wandb.log(log_dict, step=i)

        tqdm.write(
            # SAC losses and JSD
            f"States: {(i + 1) * BS}, Critic: {qf1_loss:.3f}/{qf2_loss:.3f}, Policy: {policy_loss:.3f}, Alpha: {alpha:.3f}, JSD: {jsd:.4f}, Replay: {len(memory)}"
        )

if USE_WANDB:
    wandb.finish()

# Save model and arguments as JSON
save_path = os.path.join("saved_models", run_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    torch.save(sac_agent.policy.state_dict(), os.path.join(save_path, "model.pt"))
    torch.save(bw_model.state_dict(), os.path.join(save_path, "bw_model.pt"))
    torch.save(sac_agent.critic.state_dict(), os.path.join(save_path, "critic.pt"))
    torch.save(sac_agent.critic_target.state_dict(), os.path.join(save_path, "critic_target.pt"))
    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(vars(args), f)
