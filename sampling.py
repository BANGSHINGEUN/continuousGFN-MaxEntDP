import torch
import numpy as np
import random

def sample_actions(env, model, states):
    # states is a tensor of shape (n, dim)
    batch_size = states.shape[0]
    out = model.to_dist(states)
    if isinstance(out, tuple):  # s0 input returns (dist_r, dist_theta)
        dist_r, dist_theta = out
        if model.uniform:
            samples_r = torch.rand(batch_size, device=env.device)
            samples_theta = torch.rand(batch_size, device=env.device)
        else:
            samples_r = dist_r.sample(torch.Size((batch_size,)))
            samples_theta = dist_theta.sample(torch.Size((batch_size,)))

        actions = (
            torch.stack(
                [
                    samples_r * torch.cos(torch.pi / 2.0 * samples_theta),
                    samples_r * torch.sin(torch.pi / 2.0 * samples_theta),
                ],
                dim=1,
            )
            * env.delta
        )

        return actions, samples_r, samples_theta
    else:
        dist = out

        # Automatic termination: check if min_i(1 - state_i) <= env.delta
        # This means at least one dimension is within delta of the boundary
        should_terminate = torch.any(states >= 1 - env.delta, dim=-1)

        A = torch.where(
            states[:, 0] <= 1 - env.delta,
            0.0,
            2.0 / torch.pi * torch.arccos((1 - states[:, 0]) / env.delta),
        )
        B = torch.where(
            states[:, 1] <= 1 - env.delta,
            1.0,
            2.0 / torch.pi * torch.arcsin((1 - states[:, 1]) / env.delta),
        )
        assert torch.all(
            B[~should_terminate] >= A[~should_terminate]
        )
        if model.uniform:
            samples = torch.rand(batch_size, device=env.device)
        else:
            samples = dist.sample()

        actions = samples * (B - A) + A
        actions *= torch.pi / 2.0
        actions = (
            torch.stack([torch.cos(actions), torch.sin(actions)], dim=1) * env.delta
        )

        # Set terminal actions and zero logprobs for terminated states
        actions[should_terminate] = -float("inf")
        samples[should_terminate] = -float("inf")
    return actions, samples


def sample_trajectories(env, model, n_trajectories):
    states = torch.zeros((n_trajectories, env.dim), device=env.device)
    actionss = []
    sampless = []
    trajectories = [states]
    first = True
    while not torch.all(states == env.sink_state):
        non_terminal_mask = torch.all(states != env.sink_state, dim=-1)
        actions = torch.full(
            (n_trajectories, env.dim), -float("inf"), device=env.device
        )
        samples = torch.full(
            (n_trajectories, env.dim), -float("inf"), device=env.device
        )
        if first:
            first = False
            non_terminal_actions, non_terminal_samples_r, non_terminal_samples_theta = sample_actions(
                env,
                model,
                states[non_terminal_mask],
            )
            non_terminal_samples = torch.cat([non_terminal_samples_r.unsqueeze(-1), non_terminal_samples_theta.unsqueeze(-1)], dim=-1)
            samples[non_terminal_mask] = non_terminal_samples.reshape(-1, env.dim)
        else:
            non_terminal_actions, non_terminal_samples = sample_actions(
                env,
                model,
                states[non_terminal_mask],
            )
            non_terminal_samples = torch.cat([non_terminal_samples.unsqueeze(-1), torch.zeros_like(non_terminal_samples).unsqueeze(-1)], dim=-1)
            samples[non_terminal_mask] = non_terminal_samples.reshape(-1, env.dim)
        
        actions[non_terminal_mask] = non_terminal_actions.reshape(-1, env.dim)
        actionss.append(actions)
        sampless.append(samples)
        states = env.step(states, actions)
        trajectories.append(states)
    trajectories = torch.stack(trajectories, dim=1)
    actionss = torch.stack(actionss, dim=1)
    sampless = torch.stack(sampless, dim=1)
    return trajectories, actionss, sampless


def evaluate_forward_step_logprobs(env, model, current_states, samples):
    if torch.all(current_states[0] == 0.0):
        dist_r, dist_theta = model.to_dist(current_states)
        samples_r = samples[:, 0]
        samples_theta = samples[:, 1]

        step_logprobs = (
            dist_r.log_prob(samples_r)
            + dist_theta.log_prob(samples_theta)
            - torch.log(samples_r * env.delta)
            - np.log(np.pi / 2)
            - np.log(env.delta)  # why ?
            )
        
    else:
        step_logprobs = torch.zeros((current_states.shape[0],), device=env.device)
        should_terminate = torch.any(samples == -float("inf"), dim=-1)
        if current_states.shape[0] == should_terminate.sum():
            all_terminate = torch.all(samples == -float("inf"), dim=-1)
            step_logprobs[all_terminate] = -float("inf")
            return step_logprobs
        non_terminal_states = current_states[~should_terminate]
        non_terminal_samples = samples[~should_terminate]
        dist = model.to_dist(non_terminal_states)

        A = torch.where(
            non_terminal_states[:, 0] <= 1 - env.delta,
            0.0,
            2.0 / torch.pi * torch.arccos((1 - non_terminal_states[:, 0]) / env.delta),
        )
        B = torch.where(
            non_terminal_states[:, 1] <= 1 - env.delta,
            1.0,
            2.0 / torch.pi * torch.arcsin((1 - non_terminal_states[:, 1]) / env.delta),
        )

        non_terminal_step_logprobs = (
            dist.log_prob(non_terminal_samples[:,0])
            - np.log(env.delta)
            - np.log(np.pi / 2)
            - torch.log(B - A)
        )
        step_logprobs[~should_terminate] = non_terminal_step_logprobs
        all_terminate = torch.all(samples == -float("inf"), dim=-1)
        step_logprobs[all_terminate] = -float("inf")

    return step_logprobs

def evaluate_forward_logprobs(env, model, trajectories, sampless):
    logprobs = torch.zeros((trajectories.shape[0],), device=env.device)
    all_logprobs = []
    for i in range(trajectories.shape[1] - 1):
        current_states = trajectories[:, i]
        samples = sampless[:, i]
        step_logprobs = evaluate_forward_step_logprobs(env, model, current_states, samples)
        is_finite = torch.isfinite(step_logprobs)
        logprobs[is_finite] += step_logprobs[is_finite]
        all_logprobs.append(step_logprobs)
    all_logprobs = torch.stack(all_logprobs, dim=1)

    return logprobs, all_logprobs


def evaluate_backward_step_logprobs(env, model, current_states, previous_states):
    difference_1 = current_states[:, 0] - previous_states[:, 0]
    difference_1.clamp_(min=0.0, max=env.delta)  # 안전한 클램프

    x = current_states[:, 0]
    y = current_states[:, 1]
    delta = env.delta

    # 기본값: 내부 영역 (δ ≤ x' ≤ 1-δ, δ ≤ y' ≤ 1-δ) 에서는 A = 0, B = 1
    A = torch.zeros_like(x)
    B = torch.ones_like(y)

    # 원 중심과 거리 제곱
    # 원1: 중심 (1-δ, 0)
    d1_sq = (x - (1.0 - delta)) ** 2 + y ** 2
    # 원2: 중심 (0, 1-δ)
    d2_sq = x ** 2 + (y - (1.0 - delta)) ** 2
    delta_sq = delta ** 2

    eps = 1e-6  # arcsin / arccos 수치 안정용

    # ---------- B 정의 ----------
    # ① 아래쪽 + (1-δ, 0) 근처 원 안:
    #    || x' - (1-δ), y' || ≤ δ  또는  0 ≤ x' ≤ 1-δ, 0 ≤ y' ≤ δ
    mask_B_bottom = (d1_sq <= delta_sq) | ((x <= 1.0 - delta) & (y <= delta))
    if mask_B_bottom.any():
        arg = (y[mask_B_bottom] / delta).clamp(-1.0 + eps, 1.0 - eps)
        B[mask_B_bottom] = 2.0 / torch.pi * torch.arcsin(arg)

    # ③ 오른쪽 스트립 (1-δ ≤ x' ≤ 1) 이면서 위 원 밖:
    #    1-δ ≤ x' ≤ 1, 0 ≤ y' ≤ 1,  || x' - (1-δ), y' || ≥ δ
    mask_B_right = (x >= 1.0 - delta) & (d1_sq >= delta_sq)
    if mask_B_right.any():
        arg = ((x[mask_B_right] - 1.0 + delta) / delta).clamp(-1.0 + eps, 1.0 - eps)
        B[mask_B_right] = 2.0 / torch.pi * torch.arccos(arg)

    # ---------- A 정의 ----------
    # ② 왼쪽 + (0, 1-δ) 근처 원 안:
    #    || x', y' - (1-δ) || ≤ δ  또는  0 ≤ x' ≤ δ, 0 ≤ y' ≤ 1-δ
    mask_A_left = (d2_sq <= delta_sq) | ((x <= delta) & (y <= 1.0 - delta))
    if mask_A_left.any():
        arg = (x[mask_A_left] / delta).clamp(-1.0 + eps, 1.0 - eps)
        A[mask_A_left] = 2.0 / torch.pi * torch.arccos(arg)

    # ④ 위쪽 스트립 (1-δ ≤ y' ≤ 1) 이면서 왼쪽 원 밖:
    #    0 ≤ x' ≤ 1, 1-δ ≤ y' ≤ 1,  || x', y' - (1-δ) || ≥ δ
    mask_A_top = (y >= 1.0 - delta) & (d2_sq >= delta_sq)
    if mask_A_top.any():
        arg = ((y[mask_A_top] - 1.0 + delta) / delta).clamp(-1.0 + eps, 1.0 - eps)
        A[mask_A_top] = 2.0 / torch.pi * torch.arcsin(arg)

    # ⑤ 내부 영역 (δ ≤ x' ≤ 1-δ, δ ≤ y' ≤ 1-δ) 는 A=0, B=1 이 기본값이므로
    #    따로 마스크를 줄 필요는 없음 (이미 초기값으로 세팅)

    dist = model.to_dist(current_states)

    step_logprobs = (
        dist.log_prob(
            (
                1.0
                / (B - A)
                * (2.0 / torch.pi * torch.acos(difference_1 / delta) - A)
            ).clamp(1e-4, 1 - 1e-4)
        ).clamp_max(100)
        - np.log(delta)
        - np.log(np.pi / 2)
        - torch.log(B - A)
    )

    return step_logprobs


def evaluate_backward_logprobs(env, model, trajectories):
    logprobs = torch.zeros((trajectories.shape[0],), device=env.device)
    all_logprobs = []
    for i in range(trajectories.shape[1] - 2, 1, -1):
        all_step_logprobs = torch.full(
            (trajectories.shape[0],), -float("inf"), device=env.device
        )
        non_sink_mask = torch.all(trajectories[:, i] != env.sink_state, dim=-1)
        current_states = trajectories[:, i][non_sink_mask]
        previous_states = trajectories[:, i - 1][non_sink_mask]

        step_logprobs = evaluate_backward_step_logprobs(
            env, model, current_states, previous_states
        )

        if torch.any(torch.isnan(step_logprobs)):
            raise ValueError("NaN in backward logprobs")

        if torch.any(torch.isinf(step_logprobs)):
            raise ValueError("Inf in backward logprobs")

        logprobs[non_sink_mask] += step_logprobs
        all_step_logprobs[non_sink_mask] = step_logprobs

        all_logprobs.append(all_step_logprobs)

    all_logprobs.append(torch.zeros((trajectories.shape[0],), device=env.device))
    all_logprobs = torch.stack(all_logprobs, dim=1)

    return logprobs, all_logprobs.flip(1)

def evaluate_state_flows(env, model, trajectories, logZ):
    state_flows = torch.full(
        (trajectories.shape[0], trajectories.shape[1]),
        -float("inf"),
        device=trajectories.device,
    )
    non_sink_mask = torch.all(trajectories != env.sink_state, dim=-1)
    state_flows[non_sink_mask] = model(trajectories[non_sink_mask]).squeeze(-1)
    state_flows[:, 0] = logZ

    return state_flows[:, :-1]