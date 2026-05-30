# Fix Plan for All HWs

## HW2: GRPO is broken — 3 bugs

### Bug 1: Wrong rollout batch size (CRITICAL)

The spec defines:
```
rollout_batch_size = 32  (TOTAL rollout responses)
group_size = 8
n_prompts_per_rollout_batch = rollout_batch_size // group_size = 4
```

So we should sample 4 prompts, each getting 8 rollouts = 32 total.

But `run_grpo.py` does:
```python
indices = random.sample(range(len(train_prompts)), rollout_batch_size)  # 32 PROMPTS
repeated_prompts = [p for p in batch_prompts for _ in range(group_size)]  # 256 total!
```

This generates 256 responses instead of 32. The gradient signal is diluted across
too many examples and OOM was caused by this.

FIX: Change to `n_prompts = rollout_batch_size // group_size` = 4 prompts.

### Bug 2: vLLM weights never sync (MINOR for 50 steps)

vLLM loads the initial Qwen model once and never updates. After each gradient
step the policy drifts slightly from what vLLM generates with. However, since
we take epochs_per_rollout_batch=1 (one gradient step per batch), the staleness
is exactly 1 step. For 50 total steps with lr=1e-5, the drift is small.

This is acceptable per the spec design (on-policy with 1 step). The staff 
implementation likely also uses vLLM without syncing. NOT the primary bug.

The real fix if needed: reload vLLM from saved weights every 10 steps.
For now: leave as-is. The batch size fix (Bug 1) is what matters.

### Bug 3: Gradient accumulation math is wrong

With 256 responses and hf_chunk_size=4, we get 64 microbatches.
Each divides loss by gradient_accumulation_steps=16.
Effective: 64 backward passes, each dividing by 16 = effective batch divisor of 16,
but accumulating over 64 chunks = 4x too large gradient.

With the fix to Bug 1 (32 responses), use micro_train_batch_size = 2,
gradient_accumulation_steps = 16, giving 16 microbatches of 2. Each divides by 16. Correct.

### Expected behavior after fix:
- Staff achieves 0.68 average validation reward in 50 steps (~1 hour)
- Our previous run: reward stayed ~0.04-0.07. After fix, expect 0.4-0.7

---

## HW3: VLM 0% accuracy — 1 critical bug + 1 weakness

### Bug 1: generate() decoding includes prompt tokens (CRITICAL)

In `vlm/model.py:generate()`, line 189:
```python
outputs = self.decoder.generate(inputs_embeds=inputs_embeds, ...)
generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

When `inputs_embeds` is passed to HF generate(), the output tensor includes
positions for the input embeddings (filled with dummy/BOS token IDs).
`batch_decode` decodes ALL of these, producing text like:
"<pad><pad>...Question: How many? Answer: 3"

But evaluation does:
```python
correct = pred.strip().lower() == gold.strip().lower()
```

The pred includes the full prompt text, so it NEVER matches "3".

FIX: Slice output to only decode newly generated tokens:
```python
input_len = inputs_embeds.shape[1]
generated_ids = outputs[:, input_len:]
generated = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

### Weakness: LoRA accuracy (0.41) is low but maybe acceptable

CLIP was trained on EuroSAT (10 classes, satellite), RESISC45 is 45 classes
of a different remote sensing distribution. With only 10 epochs and small ViT
(patch_size=8, d_model=384, img_size=64), domain shift causes low accuracy.
The numbers are internally consistent (LP < LoRA < FT) so the writeup framing is OK.

To improve: run more epochs (20-30) or use larger learning rate for LoRA.
Not a code bug, just weak results from short training.

---

## HW4: Code is correct, writeup is incomplete

### What's missing from writeup (NO retraining needed):
- Problem 5.A.i: VP drift coefficient f(x,t) = -1/2 * beta(t) * x
- Problem 5.A.ii: VP diffusion coefficient g(t) = sqrt(beta(t))
  (These are just written answers from the SDE definition)

### What needs GPU (sample generation):
- Problem 5.C: Dataset visualization, EM sample grid, PC sample grid, training curves
- Problem 6.B: Generate samples at various step counts for comparison
- Problem 6.D: 4x8 qualitative grid comparing VP/RectFlow/Reflow

These require loading trained models and generating ~100 images each.
~20 minutes on A100 total.

### What's needed for writeup enrichment:
- Add training curves for RectFlow and Reflow
- Add sample quality comparison discussion
- The existing KID table is solid and the proofs are complete

---

## HW1: DONE, no fixes needed
- Val loss 1.80, beats target of 2.0
- Ablations complete, figures included
- Code is correct

---

## RunPod Execution Plan

Spin 1x A100-SXM pod. Total time estimate: ~2 hours.

### Phase 1: HW2 GRPO fix (60 min)
1. Fix run_grpo.py:
   - n_prompts = 4 (not 32)
   - Fix micro batch math: 16 microbatches of 2
   - Add vLLM weight sync after each step (save + reload model in LLM engine)
2. Run normalize_by_std=True for 50 steps
3. Run normalize_by_std=False for 50 steps
4. Generate new grpo_curves.pdf plot

### Phase 2: HW3 VLM fix (30 min)
1. Fix model.py generate() to slice outputs correctly
2. Re-run eval on existing checkpoints (if they exist on pod)
3. If no checkpoints: retrain projector 2000 steps (fast, ~15 min)
4. Report corrected accuracy numbers

### Phase 3: HW4 sample generation (20 min)
1. Load VP best.pt, generate EM and PC sample grids
2. Load RectFlow best.pt, generate sample grid
3. Load Reflow best.pt, generate sample grid
4. Create 4x8 comparison figure
5. Plot RectFlow/Reflow training curves from saved .npy files

### Phase 4: Update writeups and compile (10 min)
1. Update hw2/writeup.tex with new GRPO results
2. Update hw3/writeup.tex with corrected VLM accuracy
3. Add missing written answers to hw4/writeup.tex
4. Add sample figures to hw4/writeup.tex
5. Compile all PDFs with tectonic
6. Push to GitHub
