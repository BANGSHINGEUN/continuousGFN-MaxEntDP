"""
Debug trajectories_to_transitions to see how it processes data
"""
import torch
import numpy as np
from env import Box, get_last_states
from sac import SAC
from model import CirclePB
from sampling import sample_trajectories, evaluate_backward_logprobs
from sac_replay_memory import trajectories_to_transitions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--delta", type=float, default=0.25)
parser.add_argument("--env_epsilon", type=float, default=1e-10)
parser.add_argument("--n_components", type=int, default=2)
parser.add_argument("--n_components_s0", type=int, default=4)
parser.add_argument("--beta_min", type=float, default=0.1)
parser.add_argument("--beta_max", type=float, default=5.0)
parser.add_argument("--PB", type=str, default="uniform")
parser.add_argument("--gamma_scheduler", type=float, default=0.5)
parser.add_argument("--scheduler_milestone", type=int, default=2500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--BS", type=int, default=3)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--sac_alpha", type=float, default=0.2)
parser.add_argument("--automatic_entropy_tuning", action="store_true", default=False)
parser.add_argument("--target_update_interval", type=int, default=1)
parser.add_argument("--Critic_hidden_size", type=int, default=256)
parser.add_argument("--replay_size", type=int, default=1000000)
parser.add_argument("--sac_batch_size", type=int, default=256)
parser.add_argument("--updates_per_step", type=int, default=1)
parser.add_argument("--reward_type", type=str, default="baseline")
parser.add_argument("--reward_debug", action="store_true", default=False)

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = args.device
env = Box(dim=args.dim, delta=args.delta, epsilon=args.env_epsilon, device_str=device, reward_type=args.reward_type, reward_debug=args.reward_debug)
sac_agent = SAC(args, env)
bw_model = CirclePB(hidden_dim=args.hidden_dim, n_hidden=args.n_hidden, torso=sac_agent.policy.torso if args.PB == "tied" else None, uniform=args.PB == "uniform", n_components=args.n_components, beta_min=args.beta_min, beta_max=args.beta_max).to(device)

print("="*80)
print("SAMPLING TRAJECTORIES")
print("="*80)

trajectories, actionss, logprobs, all_logprobs = sample_trajectories(env, sac_agent.policy, args.BS)
last_states = get_last_states(env, trajectories)
logrewards = env.reward(last_states).log()
bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(env, bw_model, trajectories)

print(f"\nBatch size: {args.BS}")
print(f"Trajectories shape: {trajectories.shape}")
print(f"Actions shape: {actionss.shape}")
print(f"Backward logprobs shape: {all_bw_logprobs.shape}")

for traj_idx in range(args.BS):
    print("\n" + "="*80)
    print(f"TRAJECTORY {traj_idx}")
    print("="*80)
    traj = trajectories[traj_idx]
    actions = actionss[traj_idx]
    bw_lps = all_bw_logprobs[traj_idx]
    non_sink = torch.all(traj != env.sink_state, dim=-1)
    length = non_sink.sum().item()
    print(f"\nLength: {length}, BW logprobs: {len(bw_lps)}")
    print("\nSteps:")
    for t in range(min(length + 2, len(traj))):
        state = traj[t]
        is_sink = torch.all(state == env.sink_state).item()
        if t < len(actions):
            action = actions[t]
            is_exit = torch.all(torch.isinf(action)).item()
        else:
            action, is_exit = None, None
        bw_lp = bw_lps[t].item() if t < len(bw_lps) else None
        status = "SINK" if is_sink else ("s0" if t == 0 else "")
        print(f"  t={t}: s={state.cpu().numpy() if not is_sink else 'SINK':30s} ", end="")
        if action is not None:
            print(f"a={'EXIT' if is_exit else str(action.cpu().numpy()):20s} ", end="")
        print(f"bw={bw_lp if bw_lp else 'N/A':7} {status}")
    print(f"Last state: {last_states[traj_idx].cpu().numpy()}, Log reward: {logrewards[traj_idx].item():.4f}")

print("\n" + "="*80)
print("CONVERTING TO TRANSITIONS")
print("="*80)

all_states, all_actions, all_rewards, all_next_states, all_dones = trajectories_to_transitions(trajectories, actionss, all_bw_logprobs, last_states, logrewards, env)

print(f"\nTotal: {len(all_states)} transitions")
intermediate_mask = all_dones.squeeze() == 0
terminal_mask = all_dones.squeeze() == 1
print(f"  Intermediate (done=0): {intermediate_mask.sum().item()}")
print(f"  Terminal (done=1): {terminal_mask.sum().item()}")

print("\nFirst 3 intermediate:")
for idx in torch.where(intermediate_mask)[0][:3]:
    s, a, r, s_next, done = all_states[idx], all_actions[idx], all_rewards[idx], all_next_states[idx], all_dones[idx]
    print(f"  s={s.cpu().numpy()} a={a.cpu().numpy()} r={r.item():.4f} s'={s_next.cpu().numpy()} done={done.item()}")

print("\nAll terminal:")
for idx in torch.where(terminal_mask)[0]:
    s, a, r, s_next, done = all_states[idx], all_actions[idx], all_rewards[idx], all_next_states[idx], all_dones[idx]
    is_sink = torch.all(s_next == env.sink_state).item()
    print(f"  s={s.cpu().numpy()} a={a.cpu().numpy()} r={r.item():.4f} s'={'SINK' if is_sink else s_next.cpu().numpy()} done={done.item()}")

print("\n✓ Done")
