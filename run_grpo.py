"""GRPO training — uses HF model for both rollouts and training so weights stay in sync."""
import json
import os
import random
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "online"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, '/root/hw2')

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.eval import load_gsm8k_examples, build_prompts
from alignment.grpo import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    compute_grpo_clip_loss,
    tokenize_prompt_and_output,
)
from alignment.rewards import answer_tag_reward_fn
from alignment.prompts import COT_PROMPT_TEMPLATE

HF_MODEL = "Qwen/Qwen2.5-Math-1.5B"
ARTIFACTS = Path("/root/hw2/artifacts")
ARTIFACTS.mkdir(exist_ok=True)


def generate_rollouts(model, tokenizer, prompts, max_new_tokens=256, temperature=1.0,
                      batch_size=8, device="cuda"):
    """Generate responses using the current policy weights."""
    model.eval()
    all_responses = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                min_new_tokens=4,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            input_len = enc.input_ids.shape[1]
            for j in range(len(batch)):
                gen_ids = out[j, input_len:]
                all_responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    model.train()
    return all_responses


def eval_accuracy(model, tokenizer, prompts, ground_truths, device="cuda"):
    responses = generate_rollouts(model, tokenizer, prompts, max_new_tokens=256,
                                   temperature=0.0, batch_size=8, device=device)
    correct = sum(
        answer_tag_reward_fn(r, g)["answer_reward"]
        for r, g in zip(responses, ground_truths)
    )
    return correct / len(ground_truths)


def safe_grpo_step(policy_log_probs, response_mask, gradient_accumulation_steps,
                   advantages, old_log_probs, cliprange):
    per_token_loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    mask_float = response_mask.to(per_token_loss.dtype)
    response_lengths = response_mask.sum(dim=1).clamp(min=1)
    masked_loss = torch.nan_to_num(per_token_loss, nan=0.0) * mask_float
    per_example_loss = masked_loss.sum(dim=1) / response_lengths
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def run_grpo(normalize_by_std: bool, n_steps: int, tag: str):
    print(f"\n=== GRPO Training: {tag} normalize_by_std={normalize_by_std} ===")
    device = "cuda"

    run = wandb.init(project="cs148b-hw2", name=tag, config={
        "model": HF_MODEL, "n_steps": n_steps, "normalize_by_std": normalize_by_std,
        "lr": 1e-5, "rollout_batch_size": 32, "group_size": 8,
        "gradient_accumulation_steps": 16, "cliprange": 1.0,
    })

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # required for decoder-only generation

    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=torch.bfloat16,
                                                  attn_implementation="flash_attention_2").to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.95))

    train_examples = load_gsm8k_examples("train")
    train_prompts = build_prompts(train_examples, str(COT_PROMPT_TEMPLATE))
    train_gts = [ex["answer"].split("#### ")[-1].strip() for ex in train_examples]

    eval_examples = load_gsm8k_examples("test")[:256]
    eval_prompts = build_prompts(eval_examples, str(COT_PROMPT_TEMPLATE))
    eval_gts = [ex["answer"].split("#### ")[-1].strip() for ex in eval_examples]

    n_prompts_per_batch = 4   # 4 prompts * 8 rollouts = 32 total
    group_size = 8
    gradient_accumulation_steps = 16
    cliprange = 1.0
    advantage_eps = 1e-6
    hf_chunk_size = 2
    all_metrics = []

    for step in range(n_steps):
        indices = random.sample(range(len(train_prompts)), n_prompts_per_batch)
        batch_prompts = [train_prompts[i] for i in indices]
        batch_gts_step = [train_gts[i] for i in indices]

        repeated_prompts = [p for p in batch_prompts for _ in range(group_size)]
        repeated_gts = [g for g in batch_gts_step for _ in range(group_size)]

        # Generate from CURRENT policy weights
        rollout_responses = generate_rollouts(model, tokenizer, repeated_prompts,
                                               max_new_tokens=256, temperature=1.0,
                                               batch_size=8, device=device)
        rollout_responses = [r if r.strip() else " " for r in rollout_responses]

        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=answer_tag_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )

        tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)
        advantages_dev = advantages.to(device)

        # Compute old log-probs from current policy (= rollout policy, on-policy)
        model.eval()
        old_lp_chunks = []
        with torch.no_grad():
            for i in range(0, len(repeated_prompts), hf_chunk_size):
                r = get_response_log_probs(model, input_ids[i:i + hf_chunk_size],
                                           labels[i:i + hf_chunk_size])
                old_lp_chunks.append(r["log_probs"])
        old_lp = torch.cat(old_lp_chunks, dim=0).detach()
        model.train()
        torch.cuda.empty_cache()

        # Gradient accumulation — policy logprobs now computed AFTER old_lp is frozen
        optimizer.zero_grad()
        total_loss = 0.0
        for start in range(0, len(repeated_prompts), hf_chunk_size):
            end = min(start + hf_chunk_size, len(repeated_prompts))
            result = get_response_log_probs(model, input_ids[start:end], labels[start:end])
            mb_adv = advantages_dev[start:end].unsqueeze(-1)
            loss, _ = safe_grpo_step(
                policy_log_probs=result["log_probs"],
                response_mask=response_mask[start:end],
                gradient_accumulation_steps=gradient_accumulation_steps,
                advantages=mb_adv,
                old_log_probs=old_lp[start:end],
                cliprange=cliprange,
            )
            total_loss += loss.item()
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        metrics = {
            "step": step, "loss": total_loss,
            "mean_reward": reward_meta["mean_reward"],
            "std_reward": reward_meta["std_reward"],
        }
        all_metrics.append(metrics)
        run.log(metrics)
        print(f"[{tag}] step={step}/{n_steps} loss={total_loss:.6f} "
              f"mean_reward={reward_meta['mean_reward']:.4f}", flush=True)

        if (step + 1) % 10 == 0 or step == n_steps - 1:
            acc = eval_accuracy(model, tokenizer, eval_prompts, eval_gts, device=device)
            run.log({"eval_accuracy": acc, "step": step})
            print(f"[{tag}] step={step} eval_accuracy={acc:.4f}", flush=True)

    out_path = ARTIFACTS / f"{tag}_results.json"
    out_path.write_text(json.dumps({"metrics": all_metrics, "tag": tag}, indent=2))

    model_path = ARTIFACTS / tag
    model_path.mkdir(exist_ok=True)
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    run.finish()
    print(f"Done: {tag}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "normalize"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    if mode == "normalize":
        run_grpo(normalize_by_std=True, n_steps=n_steps, tag="grpo_normalize_std_v2")
    else:
        run_grpo(normalize_by_std=False, n_steps=n_steps, tag="grpo_no_normalize_std_v2")
