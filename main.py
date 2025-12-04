import json
import os
from pickle import TRUE
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
    evaluate_state_flows,
)

from utils import (
    fit_kde,
    plot_reward,
    sample_from_reward,
    plot_samples,
    estimate_jsd,
    plot_trajectories,
)

import config

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
    choices=["baseline", "ring", "angular_ring", "multi_ring", "curve", "gaussian_mixture", "corner_squares", "two_corners", "edge_boxes", "edge_boxes_corner_squares"],
    default=config.REWARD_TYPE,
    help="Type of reward function to use. To modify reward-specific parameters (radius, sigma, etc.), edit rewards.py"
)
parser.add_argument("--R0", type=float, default=config.R0, help="Baseline reward value")
parser.add_argument("--R1", type=float, default=config.R1, help="Medium reward value (e.g., outer square)")
parser.add_argument("--R2", type=float, default=config.R2, help="High reward value (e.g., inner square)")
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
)
parser.add_argument("--loss", type=str, choices=["tb", "db"], default=config.LOSS)
parser.add_argument("--gamma_scheduler", type=float, default=config.GAMMA_SCHEDULER)
parser.add_argument("--scheduler_milestone", type=int, default=config.SCHEDULER_MILESTONE)
parser.add_argument("--seed", type=int, default=config.SEED)
parser.add_argument("--lr", type=float, default=config.LR)
parser.add_argument("--lr_Z", type=float, default=config.LR_Z)
parser.add_argument("--lr_F", type=float, default=config.LR_F)
parser.add_argument("--alpha", type=float, default=config.ALPHA)
parser.add_argument("--tie_F", action="store_true", default=config.TIE_F)
parser.add_argument("--BS", type=int, default=config.BS)
parser.add_argument("--n_iterations", type=int, default=config.N_ITERATIONS)
parser.add_argument("--n_evaluation_interval", type=int, default=config.N_EVALUATION_INTERVAL)
parser.add_argument("--n_logging_interval", type=int, default=config.N_LOGGING_INTERVAL)
parser.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM)
parser.add_argument("--n_hidden", type=int, default=config.N_HIDDEN)
parser.add_argument("--n_evaluation_trajectories", type=int, default=config.N_EVALUATION_TRAJECTORIES)
parser.add_argument("--no_plot", action="store_true", default=config.NO_PLOT)
parser.add_argument("--no_wandb", action="store_true", default=config.NO_WANDB)
parser.add_argument("--wandb_project", type=str, default=config.WANDB_PROJECT)
parser.add_argument("--uniform_ratio", type=float, default=config.UNIFORM_RATIO)
parser.add_argument("--replay_size", type=int, default=config.REPLAY_SIZE)
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
lr_Z = args.lr_Z
lr_F = args.lr_F
n_iterations = args.n_iterations
BS = args.BS
n_components = args.n_components
n_components_s0 = args.n_components_s0

if seed == 0:
    seed = np.random.randint(int(1e6))

run_name = f"GFN_d{delta}_{args.reward_type}_{args.loss}_PB{args.PB}_lr{lr}_lrZ{lr_Z}_sd{seed}"
run_name += f"_replay_size{args.replay_size}"
run_name += f"_UR{args.uniform_ratio}"
run_name += f"_R0,R1,R2_{args.R0},{args.R1},{args.R2}"
run_name += f"_BS{BS}"
run_name += f"_device{device}"
print(run_name)
if USE_WANDB:
    wandb.run.name = run_name  # type: ignore

torch.manual_seed(seed)
np.random.seed(seed)

print(f"Using device: {device}")

env = Box(
    dim=dim,
    delta=delta,
    device_str=device,
    reward_type=args.reward_type,
    reward_debug=args.reward_debug,
    R0=args.R0,
    R1=args.R1,
    R2=args.R2,
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


model = CirclePF(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    n_components=n_components,
    n_components_s0=n_components_s0,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
).to(device)

bw_model = CirclePB(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    torso=model.torso if args.PB == "tied" else None,
    uniform=args.PB == "uniform",
    n_components=n_components,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
).to(device)

if args.loss == "db":
    flow_model = NeuralNet(
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        torso=None if not args.tie_F else model.torso,
        output_dim=1,
    ).to(device)

logZ = torch.zeros(1, requires_grad=True, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if args.PB != "uniform":
    optimizer.add_param_group(
        {
            "params": bw_model.output_layer.parameters()
            if args.PB == "tied"
            else bw_model.parameters(),
            "lr": lr,
        }
    )
optimizer.add_param_group({"params": [logZ], "lr": lr_Z})

if args.loss == "db":
    optimizer.add_param_group(
        {
            "params": flow_model.output_layer.parameters()
            if args.tie_F
            else flow_model.parameters(),
            "lr": lr_F,
        }
    )
    print("using flow model")

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[i * args.scheduler_milestone for i in range(1, 10)],
    gamma=args.gamma_scheduler,
)

jsd = float("inf")

for i in trange(n_iterations):
    optimizer.zero_grad()

    trajectories, actionss, logprobs, all_logprobs = sample_trajectories(
        env,
        model,
        BS,
    )
    last_states = get_last_states(env, trajectories)
    logrewards = env.reward(last_states).log()
    bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(
        env, bw_model, trajectories
    )

    # TB (Trajectory Balance) loss
    if args.loss == "tb":
        loss = torch.mean((logZ + logprobs - bw_logprobs - logrewards) ** 2)
        
    elif args.loss == "db":
        log_state_flows = evaluate_state_flows(env, flow_model, trajectories, logZ)  # type: ignore
        db_preds = all_logprobs + log_state_flows
        db_targets = all_bw_logprobs + log_state_flows[:, 1:]
        if args.alpha == 1.0:
            db_targets = torch.cat(
                [
                    db_targets,
                    torch.full(
                        (db_targets.shape[0], 1),
                        -float("inf"),
                        device=db_targets.device,
                    ),
                ],
                dim=1,
            )
            infinity_mask = db_targets == -float("inf")
            _, indices_of_first_inf = torch.max(infinity_mask, dim=1)
            db_targets = db_targets.scatter(
                1, indices_of_first_inf.unsqueeze(1), logrewards.unsqueeze(1)
            )
            flat_db_preds = db_preds[db_preds != -float("inf")]
            flat_db_targets = db_targets[db_targets != -float("inf")]
            loss = torch.mean((flat_db_preds - flat_db_targets) ** 2)

    if torch.isinf(loss):
        raise ValueError("Infinite loss")
    loss.backward()
    # clip the gradients for bw_model
    for p in bw_model.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(-10, 10).nan_to_num_(0.0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(-10, 10).nan_to_num_(0.0)
    optimizer.step()
    scheduler.step()

    if any(
        [
            torch.isnan(list(model.parameters())[i]).any()
            for i in range(len(list(model.parameters())))
        ]
    ):
        raise ValueError("NaN in model parameters")

    if i % args.n_logging_interval == 0:
        log_dict = {
            "loss": loss.item(),
            "sqrt(logZdiff**2)": np.sqrt((np.log(env.Z) - logZ.item())**2),
            "states_visited": (i + 1) * BS,
        }

        # Evaluate JSD every 500 iterations and add to the same log
        if i % args.n_evaluation_interval == 0:
            with torch.no_grad():
                trajectories,_, _, _ = sample_trajectories(
                    env, model, args.n_evaluation_trajectories
                )
                last_states = get_last_states(env, trajectories)
                kde, fig4 = fit_kde(last_states, plot=True)
                jsd = estimate_jsd(kde, true_kde)

                log_dict["JSD"] = jsd

            if not NO_PLOT:
                colors = plt.cm.rainbow(np.linspace(0, 1, 10))
                fig1 = plot_samples(last_states[:2000].detach().cpu().numpy())
                fig2 = plot_trajectories(trajectories.detach().cpu().numpy()[:20])

                log_dict["last_states"] = wandb.Image(fig1)
                log_dict["trajectories"] = wandb.Image(fig2)
                log_dict["kde"] = wandb.Image(fig4)

        if USE_WANDB:
            wandb.log(log_dict, step=i)

        tqdm.write(
            # Loss with 3 digits of precision, logZ with 2 digits of precision, true logZ with 2 digits of precision
            # Last computed JSD with 4 digits of precision
            f"States: {(i + 1) * BS}, Loss: {loss.item():.3f}, logZ: {logZ.item():.2f}, true logZ: {np.log(env.Z):.2f}, JSD: {jsd:.4f}"
        )


if USE_WANDB:
    wandb.finish()

# Save model and arguments as JSON
save_path = os.path.join("saved_models", run_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    torch.save(bw_model.state_dict(), os.path.join(save_path, "bw_model.pt"))
    torch.save(logZ, os.path.join(save_path, "logZ.pt"))
    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(vars(args), f)
