#!/usr/bin/env python3
"""Generate episode-level Plans and chunk-level Think labels with a multimodal language model."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass
ROLE_PROMPT = "You are a 7-DoF robot arm with a parallel-jaw gripper, operating in a tabletop manipulation environment. You see the world through a single camera mounted in front of you, returning short clips of 8 RGB images ordered oldest to newest. Your gripper has two states: OPEN (fingers apart) and CLOSED (fingers together). You move in 6-DoF Cartesian delta plus a gripper command."
PLAN_INSTRUCTION = "Given the task instruction below, write the action plan as a numbered list of 3 to 9 short imperative steps that will accomplish the task. Each step should be a single concrete action (verb + object). Do NOT describe the scene or your reasoning - just output the numbered list."
PLAN_FEW_SHOT = "Example 1:\nTask: pick up the alphabet soup and place it in the basket\nPlan:\n1. approach alphabet soup\n2. close gripper\n3. lift soup\n4. move above basket\n5. lower and release\n\nExample 2:\nTask: open the top drawer and put the bowl inside\nPlan:\n1. approach drawer handle\n2. close gripper on handle\n3. pull drawer open\n4. release handle\n5. approach bowl\n6. close gripper on bowl\n7. lift and move above drawer\n8. lower bowl into drawer\n9. release"
THINK_INSTRUCTION = "You are watching an 8-frame clip of your own gripper executing the plan above. Output exactly three short tags describing the present moment, in this order:\n  Phase: <which numbered plan step is currently active>\n  Gripper: <OPEN | CLOSED>\n  Next: <the immediate motion to execute next, one short clause>\n\nPHASE PREDICTION RULES (read carefully):\n  1. PRIMARY signal: the 8 images. Look at the gripper's current action — is it approaching an object, closing on it, lifting, transporting, lowering, or releasing? Map that observed action to the corresponding numbered plan step. The Gripper and Next tags should also be read directly off the images.\n  2. SECONDARY signal: the 'Episode progress' line below gives the chunk's position in the trajectory as a percentage. This is AUXILIARY information — use it only when the images are visually ambiguous (e.g. near-static frames where gripper position barely changes), or as a consistency check on your image-based phase estimate. Rough heuristic: if the plan has K steps and progress is p%, the active step is around ceil((p / 100) × K). Trust the images first; fall back to this only when needed.\n  3. Phase must progress monotonically across chunks. If the previous chunk's tags (provided when available) reported Phase X, this chunk's Phase must be ≥ X. Never regress to an earlier step even when images look similar to the previous chunk.\n\nBe concise: 5 to 15 words per tag."
THINK_FEW_SHOT = "Example output:\nPhase: step 2 (close gripper)\nGripper: OPEN\nNext: close gripper around the bowl"


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_grid_label(idx: int, total: int) -> str:
    label = f"Image {idx + 1}"
    if idx == 0:
        label += " (oldest)"
    elif idx == total - 1:
        label += " (current)"
    return f"[{label}]"


def build_plan_messages(instruction: str, init_image: Image.Image) -> tuple:
    sys_text = ROLE_PROMPT + "\n\n" + PLAN_INSTRUCTION + "\n\n" + PLAN_FEW_SHOT
    user_blocks = [
        {"type": "text", "text": "Initial scene:"},
        {"type": "image"},
        {"type": "text", "text": f"\nTask: {instruction}\nPlan:"},
    ]
    messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": user_blocks}]
    return (messages, [init_image])


def build_think_messages(
    instruction: str,
    plan_text: str,
    images: list,
    step_idx: int,
    total_steps: int,
    prev_think: str | None = None,
    proprio: list | None = None,
) -> tuple:
    sys_text = ROLE_PROMPT + "\n\n" + THINK_INSTRUCTION + "\n\n" + THINK_FEW_SHOT
    pct = int(round(100 * step_idx / max(total_steps - 1, 1)))
    if pct < 25:
        progress_portion = "early portion"
    elif pct < 60:
        progress_portion = "middle portion"
    elif pct < 85:
        progress_portion = "late portion"
    else:
        progress_portion = "final portion"
    gripper_hint = None
    if proprio is not None and len(proprio) >= 8:
        try:
            q1 = float(proprio[6])
            q2 = float(proprio[7])
            gap = abs(q1 - q2)
            if gap < 0.02:
                gripper_hint = f"CLOSED (qpos gap={gap:.3f}m, near zero)"
            elif gap < 0.05:
                gripper_hint = f"PARTIALLY CLOSED (qpos gap={gap:.3f}m)"
            else:
                gripper_hint = f"OPEN (qpos gap={gap:.3f}m, fingers apart)"
        except (TypeError, ValueError):
            pass
    user_blocks = [
        {"type": "text", "text": f"Task: {instruction}"},
        {"type": "text", "text": f"Plan:\n{plan_text}"},
        {
            "type": "text",
            "text": f"Episode progress (auxiliary — use only if images are ambiguous): chunk {step_idx} of {total_steps} (≈{pct}% through trajectory, {progress_portion}).",
        },
    ]
    if gripper_hint is not None:
        user_blocks.append(
            {
                "type": "text",
                "text": f"Current gripper qpos reading: {gripper_hint}. Use this to set the Gripper tag accurately.",
            }
        )
    if prev_think:
        user_blocks.append(
            {
                "type": "text",
                "text": f"Previous chunk's tags (for continuity, do NOT regress to earlier steps):\n{prev_think}",
            }
        )
    user_blocks.append({"type": "text", "text": "Recent observation (8 frames, oldest to newest):"})
    for i in range(len(images)):
        user_blocks.append({"type": "text", "text": pil_grid_label(i, len(images))})
        user_blocks.append({"type": "image"})
    user_blocks.append(
        {"type": "text", "text": "\nOutput your three tags in the format shown above:"}
    )
    messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": user_blocks}]
    return (messages, images)


def clean_plan_output(text: str) -> str:
    text = text.strip()
    text = re.sub("<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub("^\\s*Plan\\s*:?\\s*\\n?", "", text, flags=re.IGNORECASE)
    text = re.split("\\n\\s*Example\\b", text)[0]
    text = re.split("\\n\\s*Task\\s*:", text)[0]
    lines = []
    for ln in text.split("\n"):
        ln = ln.rstrip()
        if not ln.strip():
            continue
        if re.match("^\\s*\\d+[\\.\\)]\\s+", ln):
            lines.append(ln.strip())
        elif lines and (not re.match("^\\s*[A-Z][a-z]+\\s*:", ln)):
            lines[-1] += " " + ln.strip()
        else:
            break
    if not lines:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()][:5]
    return "\n".join(lines)


def clean_think_output(text: str) -> str:
    text = text.strip()
    text = re.sub("<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.split("\\n\\s*Example\\b", text)[0]
    keep = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if re.match("^(Phase|Gripper|Next)\\s*:", ln, re.IGNORECASE):
            keep.append(ln)
            if len(keep) == 3:
                break
    if keep:
        return "\n".join(keep)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()][:3]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_jsonl", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
    p.add_argument(
        "--max_num_seqs",
        type=int,
        default=4,
        help="vLLM max concurrent sequences (effective batch).",
    )
    p.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    p.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Max context length in tokens. 8 images at 224^2 produce ~2k image tokens; plan + think + few-shot is ~1.5k. 8k gives generous headroom.",
    )
    p.add_argument(
        "--limit_mm_per_prompt",
        type=int,
        default=8,
        help="vLLM image cap per prompt (must be >= 8).",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Maximum number of input rows to process."
    )
    p.add_argument("--max_plan_tokens", type=int, default=180)
    p.add_argument(
        "--max_think_tokens",
        type=int,
        default=120,
        help="Generation budget for one Phase/Gripper/Next block. ~50 tokens minimum, 120 leaves headroom for slightly longer Next clauses without truncating mid-word.",
    )
    p.add_argument(
        "--thinks_per_demo",
        type=int,
        default=999,
        help="N evenly-spaced chunks per demo to label with think; 999 (default) processes all chunks. Reduce only if generation time is the bottleneck.",
    )
    p.add_argument(
        "--think_stride",
        type=int,
        default=1,
        help="Generate think every Nth chunk (frame) within a demo. stride=10 means generate at step 0, 10, 20, ...; skipped chunks inherit the most recent generated think via forward-fill. Default 1 (no skipping). Overrides --thinks_per_demo when > 1.",
    )
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Use Qwen's thinking mode and extract post-</think>.",
    )
    return p.parse_args()


def load_vllm(args):
    from transformers import AutoProcessor
    from vllm import LLM

    print(f"[{time.strftime('%H:%M:%S')}] Initializing vLLM with {args.model_id}", flush=True)
    llm_kwargs = dict(
        model=args.model_id,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt},
        dtype="bfloat16",
    )
    llm = LLM(**llm_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    print(f"[{time.strftime('%H:%M:%S')}] vLLM ready.", flush=True)
    return (llm, processor)


def vllm_generate_batched(
    llm, processor, samples: list, max_new_tokens: int, enable_thinking: bool = False
) -> list:
    from vllm import SamplingParams

    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
    requests = []
    for messages, images in samples:
        try:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        requests.append({"prompt": prompt_text, "multi_modal_data": {"image": images}})
    outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


def main():
    args = parse_args()
    image_root = Path(args.image_root)
    llm, processor = load_vllm(args)
    print(f"[{time.strftime('%H:%M:%S')}] Reading {args.in_jsonl}", flush=True)
    samples = []
    with open(args.in_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))
            if args.limit is not None and len(samples) >= args.limit:
                break
    print(f"[{time.strftime('%H:%M:%S')}] {len(samples)} samples read", flush=True)
    plan_keys_seen: dict = {}
    for idx, s in enumerate(samples):
        key = (s["suite"], s["task_id"])
        if key not in plan_keys_seen:
            plan_keys_seen[key] = idx
    plan_pairs = []
    plan_keys_ordered = list(plan_keys_seen.keys())
    for key in plan_keys_ordered:
        idx = plan_keys_seen[key]
        s = samples[idx]
        try:
            init_img = load_image(image_root / s["suite"] / s["images"][0])
        except Exception as e:
            print(f"  WARN plan {key}: {e}", flush=True)
            init_img = Image.new("RGB", (224, 224))
        plan_pairs.append(build_plan_messages(s["instruction"], init_img))
    print(f"[{time.strftime('%H:%M:%S')}] Generating {len(plan_pairs)} plans...", flush=True)
    t0 = time.time()
    plan_texts = vllm_generate_batched(
        llm,
        processor,
        plan_pairs,
        max_new_tokens=args.max_plan_tokens,
        enable_thinking=args.enable_thinking,
    )
    print(f"  plan elapsed: {time.time() - t0:.0f}s", flush=True)
    plans: dict = {}
    for k, t in zip(plan_keys_ordered, plan_texts):
        plans[k] = clean_plan_output(t)
    print(f"[{time.strftime('%H:%M:%S')}] Plan samples:", flush=True)
    for k in list(plans.keys())[:3]:
        snippet = plans[k].replace("\n", " | ")
        print(f"  {k}: {snippet[:200]}", flush=True)
    demo_step_indices = defaultdict(list)
    for idx, s in enumerate(samples):
        demo_step_indices[s["suite"], s["task_id"], s["demo_id"]].append((s["step"], idx))
    demo_picks: dict = {}
    for demo_key, step_idx_pairs in demo_step_indices.items():
        step_idx_pairs.sort(key=lambda x: x[0])
        n_chunks = len(step_idx_pairs)
        if args.think_stride > 1:
            picks_ranks = list(range(0, n_chunks, args.think_stride))
            if n_chunks - 1 not in picks_ranks and n_chunks > 0:
                picks_ranks.append(n_chunks - 1)
        elif n_chunks <= args.thinks_per_demo:
            picks_ranks = list(range(n_chunks))
        else:
            k = args.thinks_per_demo
            picks_ranks = [int(i * n_chunks / k) for i in range(k)]
        demo_picks[demo_key] = [(r, step_idx_pairs[r][1]) for r in picks_ranks]
    n_total_thinks = sum((len(v) for v in demo_picks.values()))
    max_picks = max((len(v) for v in demo_picks.values()), default=0)
    print(
        f"[{time.strftime('%H:%M:%S')}] Generating {n_total_thinks} thinks in {max_picks} sequential rounds (1 round = 1 chunk-rank batched across all demos)...",
        flush=True,
    )
    think_texts: dict = {}
    last_think_per_demo: dict = {}
    t0 = time.time()
    IMG_LOAD_BATCH = 256
    for round_i in range(max_picks):
        round_picks_for_this_round = []
        for demo_key, picks in demo_picks.items():
            if round_i >= len(picks):
                continue
            chunk_rank, sample_idx = picks[round_i]
            round_picks_for_this_round.append((demo_key, chunk_rank, sample_idx))
        if not round_picks_for_this_round:
            continue
        n_round = len(round_picks_for_this_round)
        n_subs = (n_round + IMG_LOAD_BATCH - 1) // IMG_LOAD_BATCH
        for sub_i in range(n_subs):
            sub_picks = round_picks_for_this_round[
                sub_i * IMG_LOAD_BATCH : (sub_i + 1) * IMG_LOAD_BATCH
            ]
            sub_pairs = []
            sub_sample_ids = []
            sub_demo_keys = []
            for demo_key, chunk_rank, sample_idx in sub_picks:
                s = samples[sample_idx]
                demo_pairs = demo_step_indices[demo_key]
                demo_total = len(demo_pairs)
                try:
                    imgs = [load_image(image_root / s["suite"] / r) for r in s["images"]]
                except Exception as e:
                    print(f"  WARN sample {sample_idx}: {e}", flush=True)
                    continue
                plan_text = plans.get((s["suite"], s["task_id"]), "")
                prev_think = last_think_per_demo.get(demo_key)
                sub_pairs.append(
                    build_think_messages(
                        s["instruction"],
                        plan_text,
                        imgs,
                        chunk_rank,
                        demo_total,
                        prev_think=prev_think,
                        proprio=s.get("proprio"),
                    )
                )
                sub_sample_ids.append(sample_idx)
                sub_demo_keys.append(demo_key)
            if not sub_pairs:
                continue
            texts = vllm_generate_batched(
                llm,
                processor,
                sub_pairs,
                max_new_tokens=args.max_think_tokens,
                enable_thinking=args.enable_thinking,
            )
            for sid, dkey, t in zip(sub_sample_ids, sub_demo_keys, texts):
                cleaned = clean_think_output(t)
                think_texts[sid] = cleaned
                last_think_per_demo[dkey] = cleaned
            del sub_pairs
        elapsed = time.time() - t0
        if max_picks > 0:
            eta = elapsed / (round_i + 1) * (max_picks - round_i - 1)
            print(
                f"  round {round_i + 1}/{max_picks}  ({len(think_texts)}/{n_total_thinks})  elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
    print(f"[{time.strftime('%H:%M:%S')}] Writing {args.out_jsonl}", flush=True)
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    fill_state: dict = {}
    n_written = 0
    with open(args.out_jsonl, "w") as fo:
        for idx, s in enumerate(samples):
            key_task = (s["suite"], s["task_id"])
            key_demo = (s["suite"], s["task_id"], s["demo_id"])
            s["plan_text"] = plans.get(key_task, "")
            if idx in think_texts:
                s["think_text"] = think_texts[idx]
                fill_state[key_demo] = think_texts[idx]
            else:
                s["think_text"] = fill_state.get(key_demo, "")
            fo.write(json.dumps(s, ensure_ascii=False) + "\n")
            n_written += 1
    print(
        f"[{time.strftime('%H:%M:%S')}] Done. {n_written} samples, {len(plans)} plans, {len(think_texts)} thinks generated.",
        flush=True,
    )


if __name__ == "__main__":
    main()
