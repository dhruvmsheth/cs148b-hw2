from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    full_sequences = [
        tokenizer.encode(p, add_special_tokens=False) + tokenizer.encode(o, add_special_tokens=False)
        for p, o in zip(prompt_strs, output_strs)
    ]
    max_len = max(len(s) - 1 for s in full_sequences)

    input_ids_list, labels_list, response_mask_list = [], [], []
    for p, o, seq in zip(prompt_strs, output_strs, full_sequences):
        prompt_len = len(tokenizer.encode(p, add_special_tokens=False))
        response_len = len(tokenizer.encode(o, add_special_tokens=False))
        seq_len = len(seq) - 1
        pad_len = max_len - seq_len
        pad = [tokenizer.pad_token_id] * pad_len

        input_ids_list.append(seq[:-1] + pad)
        labels_list.append(seq[1:] + pad)
        mask = [False] * (prompt_len - 1) + [True] * response_len + [False] * pad_len
        response_mask_list.append(mask)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(response_mask_list, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": token_log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    masked = tensor * mask.to(tensor.dtype)
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    rewards = [reward_fn(r, g)["reward"] for r, g in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor(rewards, dtype=torch.float32)

    grouped = raw_rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        advantages = centered / (grouped.std(dim=1, keepdim=True, unbiased=False) + advantage_eps)
    else:
        advantages = centered

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
    }
    return advantages.reshape(-1), raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1 - cliprange, 1 + cliprange)
    broadcast_advantages = advantages.expand_as(policy_log_probs)
    loss = -torch.minimum(ratios * broadcast_advantages, clipped_ratios * broadcast_advantages)
    metadata = {
        "clip_frac": ((ratios - 1.0).abs() > cliprange).float().mean(),
        "ratio_mean": ratios.mean(),
    }
    return loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    per_token_loss, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    mask_float = response_mask.to(per_token_loss.dtype)
    per_example_loss = (per_token_loss * mask_float).sum(dim=1) / response_mask.sum(dim=1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    records = []
    for i, (p, r, g, info) in enumerate(zip(prompts, responses, ground_truths, reward_infos)):
        record = {"prompt": p, "response": r, "ground_truth": g, **info}
        if token_entropies is not None:
            record["token_entropy"] = token_entropies[i]
        records.append(record)
    return records


def train_grpo(
    model,
    tokenizer,
    vllm_model,
    reward_fn: Callable,
    train_prompts: list[str],
    train_ground_truths: list[str],
    eval_prompts: list[str],
    eval_ground_truths: list[str],
    n_grpo_steps: int = 50,
    lr: float = 1e-5,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_max_tokens: int = 256,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    device: str = "cuda",
    wandb_run=None,
    output_dir=None,
) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    from vllm import SamplingParams
    import random

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))
    model.train()

    all_metrics = []

    for step in range(n_grpo_steps):
        # Sample a rollout batch of prompts
        indices = random.sample(range(len(train_prompts)), min(rollout_batch_size, len(train_prompts)))
        batch_prompts = [train_prompts[i] for i in indices]
        batch_gts = [train_ground_truths[i] for i in indices]

        # Each prompt gets group_size rollouts
        repeated_prompts = [p for p in batch_prompts for _ in range(group_size)]
        repeated_gts = [g for g in batch_gts for _ in range(group_size)]

        # Generate rollouts with vllm
        sampling_params = SamplingParams(
            temperature=sampling_temperature,
            max_tokens=sampling_max_tokens,
        )
        outputs = vllm_model.generate(repeated_prompts, sampling_params)
        rollout_responses = [o.outputs[0].text for o in outputs]

        # Compute advantages
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )

        # Tokenize
        tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)
        advantages = advantages.to(device)

        # Get old log probs (no grad)
        model.eval()
        with torch.no_grad():
            old_result = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
        old_log_probs = old_result["log_probs"].detach()
        model.train()

        # Gradient accumulation
        optimizer.zero_grad()
        total_loss = 0.0
        n_micro = len(repeated_prompts)
        micro_size = max(1, n_micro // gradient_accumulation_steps)

        for start in range(0, n_micro, micro_size):
            end = min(start + micro_size, n_micro)
            mb_input_ids = input_ids[start:end]
            mb_labels = labels[start:end]
            mb_mask = response_mask[start:end]
            mb_adv = advantages[start:end].unsqueeze(-1)
            mb_old_lp = old_log_probs[start:end]

            result = get_response_log_probs(model, mb_input_ids, mb_labels, return_token_entropy=False)
            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=result["log_probs"],
                response_mask=mb_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                advantages=mb_adv,
                old_log_probs=mb_old_lp,
                cliprange=cliprange,
            )
            total_loss += loss.item()

        optimizer.step()

        metrics = {
            "step": step,
            "loss": total_loss,
            "mean_reward": reward_meta["mean_reward"],
            "std_reward": reward_meta["std_reward"],
        }
        all_metrics.append(metrics)

        if wandb_run is not None:
            wandb_run.log(metrics)

        print(f"Step {step}: loss={total_loss:.4f} mean_reward={reward_meta['mean_reward']:.4f}")

    return {"metrics": all_metrics}
