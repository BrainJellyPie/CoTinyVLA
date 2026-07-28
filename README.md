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
