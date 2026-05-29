from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
from .rewards import majority_vote_tagged_answers


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return list(ds)


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=ex["question"]) for ex in examples]


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
) -> dict[str, Any]:
    """Generate model outputs, score them, and return serializable evaluation artifacts."""
    outputs = vllm_model.generate(list(prompts), eval_sampling_params)
    responses = [o.outputs[0].text for o in outputs]
    reward_infos = []
    for response, prompt in zip(responses, prompts):
        info = reward_fn(response, "")
        reward_infos.append(info)
    return {
        "prompts": list(prompts),
        "responses": responses,
        "reward_infos": reward_infos,
        "mean_reward": sum(r["reward"] for r in reward_infos) / len(reward_infos),
    }


def evaluate_vllm_with_gt(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    ground_truths: Sequence[str],
    eval_sampling_params,
) -> dict[str, Any]:
    """Generate and score outputs against ground truths."""
    outputs = vllm_model.generate(list(prompts), eval_sampling_params)
    responses = [o.outputs[0].text for o in outputs]
    reward_infos = [reward_fn(r, g) for r, g in zip(responses, ground_truths)]
    return {
        "prompts": list(prompts),
        "responses": responses,
        "ground_truths": list(ground_truths),
        "reward_infos": reward_infos,
        "mean_reward": sum(r["reward"] for r in reward_infos) / len(reward_infos),
        "accuracy": sum(r["answer_reward"] for r in reward_infos) / len(reward_infos),
    }


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def run_direct_baseline(output_path: Path) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    from vllm import LLM, SamplingParams
    from .rewards import answer_tag_reward_fn

    examples = load_gsm8k_examples("test")[:DEFAULT_VALIDATION_SIZE]
    prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    ground_truths = [ex["answer"].split("#### ")[-1].strip() for ex in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    results = evaluate_vllm_with_gt(llm, answer_tag_reward_fn, prompts, ground_truths, sampling_params)
    write_evaluation_results(results, output_path)
    print(f"Direct baseline accuracy: {results['accuracy']:.4f}")


def run_cot_baseline(output_path: Path) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    from vllm import LLM, SamplingParams
    from .rewards import answer_tag_reward_fn

    examples = load_gsm8k_examples("test")[:DEFAULT_VALIDATION_SIZE]
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    ground_truths = [ex["answer"].split("#### ")[-1].strip() for ex in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    results = evaluate_vllm_with_gt(llm, answer_tag_reward_fn, prompts, ground_truths, sampling_params)
    write_evaluation_results(results, output_path)
    print(f"CoT baseline accuracy: {results['accuracy']:.4f}")


def run_self_consistency_baseline(output_path: Path, k: int = 5) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    from vllm import LLM, SamplingParams
    from .rewards import answer_tag_reward_fn

    examples = load_gsm8k_examples("test")[:DEFAULT_VALIDATION_SIZE]
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    ground_truths = [ex["answer"].split("#### ")[-1].strip() for ex in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(temperature=1.0, max_tokens=512, n=k)

    outputs = llm.generate(prompts, sampling_params)
    correct = 0
    records = []
    for i, (output, gt) in enumerate(zip(outputs, ground_truths)):
        responses = [o.text for o in output.outputs]
        voted_answer = majority_vote_tagged_answers(responses)
        from .drgrpo_grader import grade
        is_correct = voted_answer is not None and grade(voted_answer, gt)
        correct += int(is_correct)
        records.append({
            "prompt": prompts[i],
            "responses": responses,
            "voted_answer": voted_answer,
            "ground_truth": gt,
            "correct": is_correct,
        })

    accuracy = correct / len(ground_truths)
    results = {"records": records, "accuracy": accuracy, "k": k}
    write_evaluation_results(results, output_path)
    print(f"Self-consistency (K={k}) accuracy: {accuracy:.4f}")


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE
