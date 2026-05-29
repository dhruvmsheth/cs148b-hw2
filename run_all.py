"""Run all HW2 experiments: benchmarks, baselines, and GRPO training."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import wandb


ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

WANDB_ENTITY = "dsheth_caltech"
WANDB_PROJECT = "cs148b-hw2"
HF_MODEL = "Qwen/Qwen2.5-Math-1.5B"
HF_REPO = "dhruvmsheth/cs148b-hw2-grpo"


def run_benchmarks():
    """Run Part 1 benchmarks."""
    from systems.benchmark import BenchmarkConfig, benchmark_model
    from systems.attention_benchmark import AttentionBenchmarkConfig, benchmark_attention_grid

    print("=== Part 1: Systems Benchmarks ===")

    bench_results = {}
    for model_size in ["small", "medium", "large"]:
        for mode in ["forward", "forward-backward"]:
            for use_bf16 in [False, True]:
                cfg = BenchmarkConfig(
                    model_size=model_size,
                    mode=mode,
                    use_bf16=use_bf16,
                    warmup_steps=5,
                    measure_steps=20,
                )
                res = benchmark_model(cfg)
                key = f"{model_size}_{mode}_bf16={use_bf16}"
                bench_results[key] = res

    # torch.compile benchmark for small model forward
    for compile_model in [True]:
        cfg = BenchmarkConfig(
            model_size="small",
            mode="forward",
            compile_model=compile_model,
            warmup_steps=5,
            measure_steps=20,
        )
        res = benchmark_model(cfg)
        bench_results[f"small_forward_compile={compile_model}"] = res

    (ARTIFACTS / "benchmark_results.json").write_text(json.dumps(bench_results, indent=2))
    print(f"Saved benchmark results to {ARTIFACTS}/benchmark_results.json")

    print("\n=== Attention Grid Benchmark ===")
    attn_cfg = AttentionBenchmarkConfig()
    attn_results = benchmark_attention_grid(attn_cfg)
    (ARTIFACTS / "attention_benchmark_results.json").write_text(json.dumps(attn_results, indent=2))
    print(f"Saved attention results to {ARTIFACTS}/attention_benchmark_results.json")

    return bench_results, attn_results


def run_gsm8k_baselines():
    """Run Part 2 GSM8K baselines."""
    from vllm import LLM, SamplingParams
    from alignment.eval import (
        load_gsm8k_examples,
        build_prompts,
        evaluate_vllm_with_gt,
        write_evaluation_results,
        DEFAULT_VALIDATION_SIZE,
    )
    from alignment.rewards import answer_tag_reward_fn
    from alignment.prompts import DIRECT_PROMPT_TEMPLATE, COT_PROMPT_TEMPLATE

    print("=== Part 2: GSM8K Baselines ===")
    llm = LLM(model=HF_MODEL, gpu_memory_utilization=0.6, max_model_len=1024)

    examples = load_gsm8k_examples("test")[:DEFAULT_VALIDATION_SIZE]
    ground_truths = [ex["answer"].split("#### ")[-1].strip() for ex in examples]

    # Direct baseline
    print("\n--- Direct Baseline ---")
    direct_prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    direct_params = SamplingParams(temperature=0.0, max_tokens=256)
    direct_results = evaluate_vllm_with_gt(llm, answer_tag_reward_fn, direct_prompts, ground_truths, direct_params)
    write_evaluation_results(direct_results, ARTIFACTS / "direct_baseline.json")
    print(f"Direct accuracy: {direct_results['accuracy']:.4f}")

    # CoT baseline
    print("\n--- CoT Baseline ---")
    cot_prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    cot_params = SamplingParams(temperature=0.0, max_tokens=512)
    cot_results = evaluate_vllm_with_gt(llm, answer_tag_reward_fn, cot_prompts, ground_truths, cot_params)
    write_evaluation_results(cot_results, ARTIFACTS / "cot_baseline.json")
    print(f"CoT accuracy: {cot_results['accuracy']:.4f}")

    # Self-consistency (K=5)
    print("\n--- Self-Consistency (K=5) ---")
    from alignment.rewards import majority_vote_tagged_answers
    from alignment.drgrpo_grader import grade

    k = 5
    sc_params = SamplingParams(temperature=1.0, max_tokens=256, n=k)
    sc_outputs = llm.generate(cot_prompts, sc_params)
    correct = 0
    sc_records = []
    for i, (output, gt) in enumerate(zip(sc_outputs, ground_truths)):
        responses = [o.text for o in output.outputs]
        voted = majority_vote_tagged_answers(responses)
        is_correct = voted is not None and grade(voted, gt)
        correct += int(is_correct)
        sc_records.append({"prompt": cot_prompts[i], "responses": responses, "voted_answer": voted, "ground_truth": gt, "correct": is_correct})
    sc_accuracy = correct / len(ground_truths)
    sc_results = {"records": sc_records, "accuracy": sc_accuracy, "k": k}
    write_evaluation_results(sc_results, ARTIFACTS / "self_consistency_k5.json")
    print(f"Self-consistency (K={k}) accuracy: {sc_accuracy:.4f}")

    return direct_results, cot_results, sc_results


def run_grpo_training(normalize_by_std: bool = True, n_steps: int = 50, run_tag: str = ""):
    """Run GRPO training on GSM8K."""
    import random
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from vllm import LLM, SamplingParams
    from alignment.eval import load_gsm8k_examples, build_prompts, evaluate_vllm_with_gt, write_evaluation_results
    from alignment.grpo import (
        tokenize_prompt_and_output,
        get_response_log_probs,
        compute_group_normalized_rewards,
        grpo_microbatch_train_step,
    )
    from alignment.rewards import answer_tag_reward_fn
    from alignment.prompts import COT_PROMPT_TEMPLATE

    tag = run_tag or ("grpo_normalize" if normalize_by_std else "grpo_no_normalize")
    print(f"\n=== GRPO Training: {tag} ===")

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=tag,
        config={
            "model": HF_MODEL,
            "n_steps": n_steps,
            "normalize_by_std": normalize_by_std,
            "lr": 1e-5,
            "rollout_batch_size": 32,
            "group_size": 8,
            "sampling_temperature": 1.0,
            "sampling_max_tokens": 256,
            "gradient_accumulation_steps": 16,
            "cliprange": 1.0,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=torch.bfloat16).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.95))

    llm = LLM(model=HF_MODEL, gpu_memory_utilization=0.25, dtype="bfloat16", max_model_len=600)

    train_examples = load_gsm8k_examples("train")
    train_prompts = build_prompts(train_examples, str(COT_PROMPT_TEMPLATE))
    train_gts = [ex["answer"].split("#### ")[-1].strip() for ex in train_examples]

    eval_examples = load_gsm8k_examples("test")[:128]
    eval_prompts = build_prompts(eval_examples, str(COT_PROMPT_TEMPLATE))
    eval_gts = [ex["answer"].split("#### ")[-1].strip() for ex in eval_examples]

    rollout_batch_size = 32
    group_size = 8
    gradient_accumulation_steps = 16
    cliprange = 1.0
    advantage_eps = 1e-6

    all_metrics = []

    for step in range(n_steps):
        indices = random.sample(range(len(train_prompts)), rollout_batch_size)
        batch_prompts = [train_prompts[i] for i in indices]
        batch_gts = [train_gts[i] for i in indices]

        repeated_prompts = [p for p in batch_prompts for _ in range(group_size)]
        repeated_gts = [g for g in batch_gts for _ in range(group_size)]

        sampling_params = SamplingParams(temperature=1.0, max_tokens=256)
        outputs = llm.generate(repeated_prompts, sampling_params)
        rollout_responses = [o.outputs[0].text for o in outputs]

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

        model.eval()
        with torch.no_grad():
            old_result = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
        old_log_probs = old_result["log_probs"].detach()
        model.train()

        optimizer.zero_grad()
        total_loss = 0.0
        n_total = len(repeated_prompts)
        micro_size = max(1, n_total // gradient_accumulation_steps)

        for start in range(0, n_total, micro_size):
            end = min(start + micro_size, n_total)
            mb_result = get_response_log_probs(
                model, input_ids[start:end], labels[start:end], return_token_entropy=False
            )
            mb_adv = advantages_dev[start:end].unsqueeze(-1)
            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=mb_result["log_probs"],
                response_mask=response_mask[start:end],
                gradient_accumulation_steps=gradient_accumulation_steps,
                advantages=mb_adv,
                old_log_probs=old_log_probs[start:end],
                cliprange=cliprange,
            )
            total_loss += loss.item()

        optimizer.step()

        metrics = {
            "step": step,
            "loss": total_loss,
            "mean_reward": reward_meta["mean_reward"],
        }
        all_metrics.append(metrics)
        run.log(metrics)
        print(f"[{tag}] Step {step}/{n_steps}: loss={total_loss:.4f} mean_reward={reward_meta['mean_reward']:.4f}")

        if (step + 1) % 10 == 0 or step == n_steps - 1:
            model.eval()
            eval_sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
            eval_results = evaluate_vllm_with_gt(llm, answer_tag_reward_fn, eval_prompts, eval_gts, eval_sampling_params)
            run.log({"eval_accuracy": eval_results["accuracy"], "step": step})
            print(f"[{tag}] Step {step} eval accuracy: {eval_results['accuracy']:.4f}")
            model.train()

    # Save final model
    model_path = ARTIFACTS / tag
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    results = {"metrics": all_metrics, "tag": tag}
    write_evaluation_results(results, ARTIFACTS / f"{tag}_results.json")

    run.finish()
    return results


def main():
    print("Starting HW2 experiments\n")

    # Part 1: Benchmarks
    try:
        bench_results, attn_results = run_benchmarks()
        print("Benchmarks complete.\n")
    except Exception as e:
        print(f"Benchmark error: {e}")

    # Part 2: Baselines
    direct_results, cot_results, sc_results = run_gsm8k_baselines()

    # Log baselines to wandb
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name="baselines")
    run.log({
        "direct_accuracy": direct_results["accuracy"],
        "cot_accuracy": cot_results["accuracy"],
        "sc_k5_accuracy": sc_results["accuracy"],
    })
    run.finish()

    # GRPO with normalize_by_std=True
    run_grpo_training(normalize_by_std=True, n_steps=50, run_tag="grpo_normalize_std")

    # GRPO with normalize_by_std=False
    run_grpo_training(normalize_by_std=False, n_steps=50, run_tag="grpo_no_normalize_std")

    print("\nAll experiments complete. Results in artifacts/")


if __name__ == "__main__":
    main()
