# CoTinyVLA

This repository provides the data preparation, supervision generation,
training, inference, and LIBERO-Plus evaluation pipeline for CoTinyVLA.

## Structure

```text
scripts/
  convert_libero_plus.py
  generate_reasoning_labels.py
  train_policy.py
  infer_libero_plus.py
  evaluate_libero_plus.py
  evaluate_libero_plus_worker.py
tests/
  test_core.py
tools/
  check_source.py
```

The evaluation code expects LIBERO-Plus to be importable as
`libero_plus.libero`. This separate namespace allows Standard LIBERO and
LIBERO-Plus to coexist in one Python environment.

<!-- RESULTS_START -->
## Results

CoTinyVLA contains approximately 0.9B parameters and is evaluated on
LIBERO-Plus and Standard LIBERO.

### LIBERO-Plus suite comparison

LIBERO-Plus contains 10,030 perturbed tasks across seven perturbation
dimensions. Each perturbed task is evaluated with one rollout.

| Model | Params | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|---:|
| OpenVLA | 7B | 19.4 | 14.0 | 15.1 | 14.3 | 15.7 |
| NORA | 3B | 47.6 | 34.4 | 38.8 | 36.3 | 39.3 |
| WorldVLA | 7B | 32.5 | 28.6 | 31.8 | 8.2 | 25.3 |
| UniVLA | 7B | 55.5 | 36.7 | 40.7 | 39.9 | 43.2 |
| π0 | 3.3B | 60.7 | 61.4 | 44.9 | 48.4 | 53.9 |
| OpenVLA-OFTw | 7B | 62.5 | 56.0 | 53.3 | 52.2 | 56.0 |
| OpenVLA-OFTm | 7B | 75.4 | 77.1 | 56.2 | 63.9 | 68.2 |
| π0-Fast | 3.3B | 74.4 | 72.7 | 57.5 | 43.4 | 62.0 |
| OpenVLA-OFT | 7B | 84.0 | 66.5 | 63.0 | 66.4 | 70.0 |
| RIPT-VLA | 7B | 85.8 | 64.3 | 58.0 | 67.5 | 68.9 |
| OpenVLA-OFT+ | 7B | 86.1 | 84.5 | 70.7 | 77.7 | 79.8 |
| **CoTinyVLA** | **0.9B** | **90.8** | **87.3** | **86.6** | **80.7** | **86.4** |

CoTinyVLA exceeds the strongest reported 7B baseline on all four
LIBERO-Plus suites while using approximately one-eighth as many
parameters.

| Suite | CoTinyVLA | Strongest baseline | Margin |
|---|---:|---:|---:|
| Spatial | 90.8 | 86.1 | **+4.7** |
| Object | 87.3 | 84.5 | **+2.8** |
| Goal | 86.6 | 70.7 | **+15.9** |
| Long | 80.7 | 77.7 | **+3.0** |

### LIBERO-Plus perturbation profile

The following table reports CoTinyVLA's success rate on each perturbation
dimension.

| Suite | Camera | Robot init. | Language | Light | Background | Noise | Layout | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial | 97.3 | 58.3 | 96.4 | 98.6 | 99.2 | 98.0 | 90.1 | **90.8** |
| Object | 99.7 | 44.0 | 89.8 | 100.0 | 100.0 | 99.5 | 85.6 | **87.3** |
| Goal | 99.3 | 73.6 | 83.7 | 99.3 | 98.2 | 98.4 | 63.1 | **86.6** |
| Long | 94.5 | 51.7 | 68.9 | 96.0 | 91.3 | 90.9 | 75.3 | **80.7** |

The largest gains over prior work occur on physical-state perturbations,
especially Robot Initial States on the Goal suite.

### Standard LIBERO comparison

Standard LIBERO uses 50 trials per task and 10 tasks per suite.

| Model | Params | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|---:|
| SmolVLA | 0.24B | 87.0 | 93.0 | 88.0 | 63.0 | 82.8 |
| CorridorVLA | 0.45B | 92.0 | 95.8 | 90.8 | 85.2 | 91.0 |
| SwiftVLA | 0.45B | 97.0 | 96.4 | 96.8 | 88.4 | 94.7 |
| VLA-0-Smol | 0.5B | 92.2 | 97.2 | 95.6 | 91.2 | 94.1 |
| Evo-1 | 0.77B | 92.7 | 97.7 | 96.3 | 92.3 | 94.8 |
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| NORA | 3B | 92.2 | 95.4 | 89.4 | 74.6 | 87.9 |
| π0 | 3.3B | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 7B | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| RIPT-VLA | 7B | 98.6 | 98.6 | 99.0 | 93.8 | 97.5 |
| **CoTinyVLA** | **0.9B** | **99.4** | **100.0** | **98.6** | **92.0** | **97.5** |

CoTinyVLA matches the strongest reported 7B average while improving over
the strongest reported sub-billion-parameter baseline by 2.7 points.

### Closed-loop inference profile

| Property | Value |
|---|---:|
| Total policy size | approximately 0.9B parameters |
| Closed-loop success | 79 / 80 episodes |
| Peak allocated GPU memory | 2.25 GiB |
| Episode-initial latency | 2.76 s |
| Steady-state latency | 1.37 s per 8-step chunk |
| Generated reasoning tokens | 26 per steady-state chunk |

The latency profile includes preprocessing, reasoning generation,
vision-language inference, the proprioception projector, and the action
head. Measurements were obtained on NVIDIA L40S GPUs with three
concurrent rollouts per GPU.

Caching the episode-level Plan preserves success while reducing
steady-state latency by 48.5% and generated reasoning tokens by 63.2%
relative to regenerating the Plan at every action chunk.

### Test-time interventions

| Intervention | Spatial | Goal | Pooled | Change |
|---|---:|---:|---:|---:|
| Unmodified policy | 100 | 100 | 100.0 | — |
| Empty Plan | 80 | 40 | 60.0 | -40.0 |
| Contradictory Plan | 75 | 35 | 55.0 | -45.0 |
| Reasoning span removed | 0 | 0 | 0.0 | -100.0 |
| Latest frame repeated | 75 | 45 | 60.0 | -40.0 |
| History order reversed | 60 | 85 | 72.5 | -27.5 |
| Wrist view occluded | 0 | 0 | 0.0 | -100.0 |
| Third-person view occluded | 100 | 95 | 97.5 | -2.5 |
| Training-set paraphrase | 100 | 100 | 100.0 | 0.0 |
| Politeness prefix | 100 | 100 | 100.0 | 0.0 |

These interventions keep the trained weights fixed and modify only the
test-time inputs or reasoning spans.
<!-- RESULTS_END -->

## Installation

```bash
python -m pip install -r requirements.txt
```

Install the labeling dependencies in the environment used for vLLM:

```bash
python -m pip install -r requirements-labeling.txt
```

## Data conversion

```bash
python scripts/convert_libero_plus.py \
  --lerobot_root /path/to/lerobot_suite \
  --suite_name libero_plus_spatial \
  --out_dir data \
  --num_images 8 \
  --chunk_size 8 \
  --gripper_to_rlds
```

## Reasoning supervision

```bash
python scripts/generate_reasoning_labels.py \
  --in_jsonl data/libero_plus_spatial/data.jsonl \
  --image_root data \
  --out_jsonl data/libero_plus_spatial/data_reasoning.jsonl \
  --model_id Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
```

## Training

```bash
torchrun --standalone --nproc_per_node=8 scripts/train_policy.py \
  --model_id Qwen/Qwen3.5-0.8B \
  --train_jsonl data/all_suites_reasoning.jsonl \
  --output_dir checkpoints/cotinyvla \
  --num_images 8 \
  --num_wrist_images 8 \
  --chunk_size 8 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --proprio_lr_mult 5 \
  --action_head_lr_mult 5 \
  --action_loss_weight 1 \
  --lm_loss_weight 0.1 \
  --attn_implementation flash_attention_2
```

Each saved checkpoint contains `student_state_dict.pt` and
`action_head_config.json` for inference.

## Inference

```bash
python scripts/infer_libero_plus.py \
  --student_dir checkpoints/cotinyvla \
  --suite libero_spatial \
  --num_trials_per_task 1 \
  --max_steps 600 \
  --attn_implementation flash_attention_2 \
  --gripper_transform normalize_only \
  --no_videos
```

## Dynamic multi-GPU evaluation

```bash
python scripts/evaluate_libero_plus.py \
  --libero_plus_root /path/to/LIBERO-plus \
  --assets_dir /path/to/LIBERO-plus-assets \
  --checkpoint_search_root checkpoints \
  --gpus 0,1,2,3,4,5,6,7 \
  --seed 7 \
  --output_dir results/libero_plus
```

The evaluator assigns tasks through a shared SQLite queue, resumes completed
work, and writes success summaries by suite, perturbation, difficulty, and task.

## Checks

```bash
python tools/check_source.py
pytest -q
```

## Model checkpoint

The final CoTinyVLA checkpoint is hosted on Hugging Face:

    euphoria-64/CoTinyVLA-Qwen3.5-0.8B

Download it with:

    hf download euphoria-64/CoTinyVLA-Qwen3.5-0.8B --local-dir checkpoints/cotinyvla
