#!/usr/bin/env python3
"""Run a trained policy on LIBERO-Plus with online Plan and Think generation."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from train_policy import (
    COT_PLAN_CLOSE,
    COT_PLAN_OPEN,
    COT_SPECIAL_TOKENS,
    COT_THINK_CLOSE,
    COT_THINK_OPEN,
    Qwen35ActionPolicy,
    build_messages_dual,
    pad_history,
)
from transformers import AutoProcessor

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("infer")

T = None
benchmark = None
get_libero_path = None
OffScreenRenderEnv = None


def load_libero_runtime() -> None:
    global T, benchmark, get_libero_path, OffScreenRenderEnv
    if T is not None:
        return
    import robosuite.utils.transform_utils as transform_utils
    from libero_plus.libero import benchmark as benchmark_module
    from libero_plus.libero import get_libero_path as path_resolver
    from libero_plus.libero.envs import OffScreenRenderEnv as render_env

    T = transform_utils
    benchmark = benchmark_module
    get_libero_path = path_resolver
    OffScreenRenderEnv = render_env


def prepare_libero_image(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ::-1])


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    a = action.copy()
    a[..., -1] = 2 * (a[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        a[..., -1] = np.sign(a[..., -1])
    return a


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    a = action.copy()
    a[..., -1] *= -1.0
    return a


def libero_dummy_action() -> list:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def load_student(
    student_dir: str,
    dtype: torch.dtype,
    attn_impl: str = "sdpa",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    model_id_override: str | None = None,
):
    with open(Path(student_dir) / "action_head_config.json") as f:
        meta = json.load(f)
    log.info(f"Student config: {meta}")
    proc_kwargs = {"trust_remote_code": True}
    if min_pixels is not None:
        proc_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        proc_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(student_dir, **proc_kwargs)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
    cot_ids = meta.get("cot_token_ids")
    uses_cot = meta.get("uses_cot", False) and cot_ids is not None
    if uses_cot:
        for name, expected_id in cot_ids.items():
            tok_str = {
                "plan_open": COT_PLAN_OPEN,
                "plan_close": COT_PLAN_CLOSE,
                "think_open": COT_THINK_OPEN,
                "think_close": COT_THINK_CLOSE,
            }[name]
            actual_id = processor.tokenizer.convert_tokens_to_ids(tok_str)
            if actual_id != expected_id:
                log.warning(
                    f"CoT token mismatch: {tok_str} → tokenizer says {actual_id}, meta says {expected_id}. Re-adding."
                )
                processor.tokenizer.add_special_tokens(
                    {"additional_special_tokens": COT_SPECIAL_TOKENS}
                )
                break
    model = Qwen35ActionPolicy(
        model_id=model_id_override or meta["model_id"],
        action_dim=meta["action_dim"],
        proprio_dim=meta["proprio_dim"],
        chunk_size=meta["chunk_size"],
        use_proprio=meta["use_proprio"],
        dtype=dtype,
        attn_impl=attn_impl,
        cot_token_ids=cot_ids if uses_cot else None,
    )
    if uses_cot:
        target_vocab = meta.get("vocab_size", len(processor.tokenizer))
        cur_vocab = model.vlm.get_input_embeddings().num_embeddings
        if cur_vocab != target_vocab:
            log.info(f"Resizing embeddings: {cur_vocab} → {target_vocab}")
            model.vlm.resize_token_embeddings(target_vocab)
    sd_path = Path(student_dir) / "student_state_dict.pt"
    log.info(f"Loading state_dict: {sd_path}")
    try:
        state_dict = torch.load(sd_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(sd_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"Missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        log.warning(f"Unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    return (model, processor, meta)


@torch.no_grad()
def predict_action_chunk(
    model: Qwen35ActionPolicy,
    processor: Any,
    pil_images: List[Image.Image],
    proprio: np.ndarray,
    instruction: str,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int = 4096,
    max_cot_tokens: int = 120,
    cot_token_ids: dict | None = None,
    return_cot_text: bool = False,
    plan_cache: dict | None = None,
    wrist_images: List[Image.Image] | None = None,
):
    wrist_images = wrist_images or []
    messages = build_messages_dual(pil_images, wrist_images, instruction)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).strip()
    use_plan_cache = (
        cot_token_ids is not None and plan_cache is not None and plan_cache.get("plan_text")
    )
    if use_plan_cache:
        cached_plan = plan_cache["plan_text"]
        plan_prefix = f"<|cot_plan|>{cached_plan}<|cot_plan_end|><|cot_think|>"
        text = text + plan_prefix
    all_images = list(pil_images) + list(wrist_images)
    inputs = processor(
        text=[text],
        images=[all_images],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(device)
    if cot_token_ids is None:
        with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            out = model(**dict(inputs), proprio=proprio_t)
        return out["pred_actions"][0].float().cpu().numpy()
    think_close_id = int(cot_token_ids["think_close"])
    gen_kwargs = dict(inputs)
    with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        gen_out = model.vlm.generate(
            **gen_kwargs,
            max_new_tokens=max_cot_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=think_close_id,
            return_dict_in_generate=True,
            use_cache=True,
        )
    full_seq = gen_out.sequences
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = full_seq[0, prompt_len:]
    eq = (new_tokens == think_close_id).nonzero(as_tuple=False).flatten()
    if eq.numel() > 0:
        new_idx = int(eq[0].item())
        emitted_think_close = True
    else:
        new_idx = new_tokens.shape[0] - 1
        emitted_think_close = False
    think_close_global_pos = prompt_len + new_idx
    target_hidden = None
    pkv = getattr(gen_out, "past_key_values", None)
    if emitted_think_close and pkv is not None:
        try:
            tc_input = torch.tensor([[think_close_id]], device=device, dtype=full_seq.dtype)
            cached_len = think_close_global_pos
            full_attn = torch.ones((1, cached_len + 1), dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                step_out = model.vlm(
                    input_ids=tc_input,
                    attention_mask=full_attn,
                    past_key_values=pkv,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                )
            target_hidden = step_out.hidden_states[-1][:, -1, :]
        except Exception as e:
            log.debug(
                f"KV-cache fast-path failed ({type(e).__name__}: {e}), falling back to re-forward."
            )
            target_hidden = None
    if target_hidden is None:
        full_attn = torch.ones_like(full_seq)
        full_seq_clipped = full_seq[:, : think_close_global_pos + 1]
        full_attn_clipped = full_attn[:, : think_close_global_pos + 1]
        rerun_kwargs = dict(
            input_ids=full_seq_clipped,
            attention_mask=full_attn_clipped,
            output_hidden_states=True,
            return_dict=True,
        )
        new_len = full_seq_clipped.shape[1]
        for key in ("pixel_values", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
            if key in inputs:
                rerun_kwargs[key] = inputs[key]
        if "mm_token_type_ids" in inputs:
            orig_mm = inputs["mm_token_type_ids"]
            if orig_mm.shape[1] < new_len:
                pad_amt = new_len - orig_mm.shape[1]
                pad = torch.zeros(
                    (orig_mm.shape[0], pad_amt), dtype=orig_mm.dtype, device=orig_mm.device
                )
                ext_mm = torch.cat([orig_mm, pad], dim=1)
            elif orig_mm.shape[1] > new_len:
                ext_mm = orig_mm[:, :new_len]
            else:
                ext_mm = orig_mm
            rerun_kwargs["mm_token_type_ids"] = ext_mm
        with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            out = model.vlm(**rerun_kwargs)
        last_hidden = out.hidden_states[-1]
        target_hidden = last_hidden[0, think_close_global_pos].unsqueeze(0)
    with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        if model.use_proprio:
            proprio_t_cast = proprio_t.to(target_hidden.dtype)
            proprio_emb = model.proprio_projector(proprio_t_cast)
            features = torch.cat([target_hidden, proprio_emb], dim=-1)
        else:
            features = target_hidden
        pred_actions_t = model.action_head(features)
    pred_actions = pred_actions_t[0].float().cpu().numpy()
    if plan_cache is not None and (not use_plan_cache):
        try:
            decoded = processor.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=False)
            a = decoded.find("<|cot_plan|>")
            b = decoded.find("<|cot_plan_end|>")
            if a >= 0 and b > a:
                plan_text = decoded[a + len("<|cot_plan|>") : b].strip()
                if plan_text:
                    plan_cache["plan_text"] = plan_text
        except Exception as e:
            log.debug(f"Plan-cache population failed: {e}")
    if return_cot_text:
        try:
            new_decoded = processor.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=False)
            if use_plan_cache:
                cot_text = f"<|cot_plan|>{plan_cache['plan_text']}<|cot_plan_end|><|cot_think|>{new_decoded}"
                end = cot_text.find("<|cot_think_end|>")
                if end >= 0:
                    cot_text = cot_text[: end + len("<|cot_think_end|>")]
            else:
                cot_text = new_decoded
                end = cot_text.find("<|cot_think_end|>")
                if end >= 0:
                    cot_text = cot_text[: end + len("<|cot_think_end|>")]
        except Exception:
            cot_text = ""
        return (pred_actions, cot_text, emitted_think_close)
    return pred_actions


def rollout_episode(
    model: Qwen35ActionPolicy,
    processor: Any,
    meta: dict,
    env,
    task_init_state: np.ndarray,
    task_desc: str,
    max_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    video_path: Path | None = None,
    gripper_transform: str = "normalize_invert",
    cot_log_path: Path | None = None,
    print_cot: bool = False,
    max_cot_tokens: int = 256,
) -> bool:
    load_libero_runtime()
    num_images = meta["num_images"]
    num_wrist_images = meta.get("num_wrist_images", 0)
    uses_dual = num_wrist_images > 0
    uses_cot = meta.get("uses_cot", False) and meta.get("cot_token_ids") is not None
    cot_log_records = []
    plan_cache: dict = {"plan_text": ""}
    env.reset()
    env.set_init_state(task_init_state)
    dummy = libero_dummy_action()
    obs = None
    for _ in range(10):
        obs, _, _, _ = env.step(dummy)
    img_buffer: deque[Image.Image] = deque(maxlen=num_images)
    wrist_buffer: deque[Image.Image] = deque(maxlen=num_wrist_images) if uses_dual else None
    first_frame = prepare_libero_image(obs["agentview_image"])
    img_buffer.append(Image.fromarray(first_frame).convert("RGB"))
    if uses_dual:
        first_wrist = prepare_libero_image(obs["robot0_eye_in_hand_image"])
        wrist_buffer.append(Image.fromarray(first_wrist).convert("RGB"))
    rollout_frames: list[np.ndarray] = [first_frame]
    done = False
    step = 0
    while not done and step < max_steps:
        proprio = np.concatenate(
            [
                obs["robot0_eef_pos"],
                T.quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        ).astype(np.float32)
        padded_third = pad_history(list(img_buffer), num_images)
        padded_wrist = pad_history(list(wrist_buffer), num_wrist_images) if uses_dual else []
        if uses_cot:
            action_chunk, cot_text, ok = predict_action_chunk(
                model,
                processor,
                padded_third,
                proprio,
                task_desc,
                device,
                dtype,
                cot_token_ids=meta["cot_token_ids"],
                return_cot_text=True,
                max_cot_tokens=max_cot_tokens,
                plan_cache=plan_cache,
                wrist_images=padded_wrist,
            )
            if not ok:
                log.warning(
                    f"[step {step}] think_close not emitted (max_cot_tokens={max_cot_tokens}); using fallback hidden."
                )
            plan_str, think_str = ("", "")
            try:
                a = cot_text.find("<|cot_plan|>")
                b = cot_text.find("<|cot_plan_end|>")
                if a >= 0 and b > a:
                    plan_str = cot_text[a + len("<|cot_plan|>") : b].strip()
                c = cot_text.find("<|cot_think|>")
                d = cot_text.find("<|cot_think_end|>")
                if c >= 0 and d > c:
                    think_str = cot_text[c + len("<|cot_think|>") : d].strip()
            except Exception:
                pass
            if print_cot:
                print(f"[step {step:3d}] PLAN:  {plan_str}", flush=True)
                print(f"[step {step:3d}] THINK: {think_str}", flush=True)
            cot_log_records.append(
                {
                    "step": int(step),
                    "plan_text": plan_str,
                    "think_text": think_str,
                    "raw_cot": cot_text,
                    "action_chunk_first": [float(x) for x in action_chunk[0]],
                }
            )
        else:
            action_chunk = predict_action_chunk(
                model,
                processor,
                padded_third,
                proprio,
                task_desc,
                device,
                dtype,
                cot_token_ids=None,
                wrist_images=padded_wrist,
            )
        action_chunk_env = np.asarray(action_chunk, dtype=np.float32)
        if gripper_transform == "normalize_invert":
            action_chunk_env = normalize_gripper_action(action_chunk_env, binarize=True)
            action_chunk_env = invert_gripper_action(action_chunk_env)
        elif gripper_transform == "normalize_only":
            action_chunk_env = normalize_gripper_action(action_chunk_env, binarize=True)
        elif gripper_transform == "none":
            pass
        else:
            raise ValueError(f"Unknown gripper_transform: {gripper_transform}")
        for a in action_chunk_env:
            obs, _, done, _ = env.step(a.tolist())
            step += 1
            frame = prepare_libero_image(obs["agentview_image"])
            rollout_frames.append(frame)
            if done:
                break
            img_buffer.append(Image.fromarray(frame).convert("RGB"))
            if uses_dual:
                wframe = prepare_libero_image(obs["robot0_eye_in_hand_image"])
                wrist_buffer.append(Image.fromarray(wframe).convert("RGB"))
    success = bool(done)
    if video_path is not None:
        try:
            import imageio

            video_path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(str(video_path), fps=30) as writer:
                for fr in rollout_frames:
                    writer.append_data(fr)
        except Exception as e:
            log.warning(f"Video save failed ({video_path}): {e}")
    if cot_log_path is not None and cot_log_records:
        try:
            cot_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cot_log_path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "_summary": True,
                            "task": task_desc,
                            "success": bool(success),
                            "n_chunks": len(cot_log_records),
                        }
                    )
                    + "\n"
                )
                for rec in cot_log_records:
                    f.write(json.dumps(rec) + "\n")
        except Exception as e:
            log.warning(f"CoT log save failed ({cot_log_path}): {e}")
    return success


SUITE_CHOICES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def eval_suite(args: argparse.Namespace) -> Dict:
    load_libero_runtime()
    dtype = (
        (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        if torch.cuda.is_available()
        else torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, meta = load_student(
        args.student_dir,
        dtype,
        args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        model_id_override=args.model_id_override,
    )
    model.to(device).eval()
    rollout_root = (
        Path(args.rollout_dir)
        if args.rollout_dir
        else Path(args.student_dir) / "rollouts" / args.suite
    )
    if not args.no_videos:
        rollout_root.mkdir(parents=True, exist_ok=True)
        log.info(f"Rollout videos → {rollout_root}")
    bm = benchmark.get_benchmark_dict()
    task_suite = bm[args.suite]()
    num_tasks = task_suite.n_tasks
    if args.task_id is not None:
        task_indices = [args.task_id]
    else:
        start = args.start_task_id if args.start_task_id is not None else 0
        end = args.end_task_id if args.end_task_id is not None else num_tasks
        task_indices = list(range(start, end))
    log.info(
        f"Suite {args.suite}: {len(task_indices)} task(s) × {args.num_trials_per_task} trials"
        + (f"  (capped at {args.max_trials} total)" if args.max_trials else "")
    )
    _nw = meta.get("num_wrist_images", 0)
    log.info(
        f"View config: third={meta['num_images']} + wrist={_nw} ({('DUAL-VIEW' if _nw > 0 else 'SINGLE-VIEW')})"
    )
    per_task: Dict[str, Dict] = {}
    total_ep, total_ok = (0, 0)
    suite_t0 = time.time()
    n_trials_done_global = 0
    for task_id in task_indices:
        if args.max_trials is not None and n_trials_done_global >= args.max_trials:
            break
        task = task_suite.get_task(task_id)
        task_desc = task.language
        task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        task_init = task_suite.get_task_init_states(task_id)
        env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        safe_desc = "".join((c if c.isalnum() or c in "-_" else "_" for c in task_desc))[:40]
        task_ok = 0
        task_t0 = time.time()
        if args.max_trials is not None:
            n_trials_this_task = min(
                args.num_trials_per_task, args.max_trials - n_trials_done_global
            )
        else:
            n_trials_this_task = args.num_trials_per_task
        for trial in range(n_trials_this_task):
            init_idx = trial % len(task_init)
            video_path = None
            if not args.no_videos:
                video_path = (
                    rollout_root
                    / f"task{task_id:02d}_{safe_desc}"
                    / f"trial{trial:02d}_pending.mp4"
                )
            cot_log_path = None
            if not args.no_cot_log and meta.get("uses_cot"):
                cot_log_root = (
                    Path(args.cot_log_dir)
                    if args.cot_log_dir
                    else Path(args.student_dir) / "cot_logs" / args.suite
                )
                cot_log_path = (
                    cot_log_root / f"task{task_id:02d}_{safe_desc}" / f"trial{trial:02d}.jsonl"
                )
            ok = rollout_episode(
                model,
                processor,
                meta,
                env,
                task_init[init_idx],
                task_desc,
                args.max_steps,
                device,
                dtype,
                video_path=video_path,
                gripper_transform=args.gripper_transform,
                cot_log_path=cot_log_path,
                print_cot=args.print_cot,
                max_cot_tokens=args.max_cot_tokens,
            )
            task_ok += int(ok)
            n_trials_done_global += 1
            if video_path is not None and video_path.exists():
                tag = "success" if ok else "failure"
                final_path = video_path.with_name(f"trial{trial:02d}_{tag}.mp4")
                video_path.replace(final_path)
        env.close()
        sr = task_ok / max(n_trials_this_task, 1)
        per_task[str(task_id)] = {
            "description": task_desc,
            "trials": n_trials_this_task,
            "successes": task_ok,
            "success_rate": sr,
        }
        total_ep += n_trials_this_task
        total_ok += task_ok
        log.info(
            f"task {task_id + 1}/{num_tasks} '{task_desc[:40]}' → {task_ok}/{n_trials_this_task} ({sr * 100:.0f}%) [{time.time() - task_t0:.0f}s]  running: {total_ok}/{total_ep}={total_ok / max(total_ep, 1) * 100:.1f}%"
        )
    overall = total_ok / max(total_ep, 1)
    log.info(
        f"Done. Overall: {total_ok}/{total_ep} = {overall * 100:.1f}%  [{time.time() - suite_t0:.0f}s]"
    )
    results = {
        "suite": args.suite,
        "student_dir": args.student_dir,
        "total_episodes": total_ep,
        "total_successes": total_ok,
        "success_rate": overall,
        "per_task": per_task,
    }
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--student_dir",
        type=str,
        required=True,
        help="Checkpoint directory containing policy weights and configuration.",
    )
    p.add_argument(
        "--model_id_override",
        type=str,
        default=None,
        help="Optional local or remote backbone path.",
    )
    p.add_argument("--suite", type=str, required=True, choices=SUITE_CHOICES)
    p.add_argument("--num_trials_per_task", type=int, default=10)
    p.add_argument(
        "--start_task_id",
        type=int,
        default=None,
        help="Resume from this task index (0-indexed, inclusive). Ignored if --task_id is set.",
    )
    p.add_argument(
        "--end_task_id",
        type=int,
        default=None,
        help="Stop before this task index (0-indexed, exclusive). Ignored if --task_id is set.",
    )
    p.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Evaluate only the specified task index.",
    )
    p.add_argument(
        "--max_trials",
        type=int,
        default=None,
        help="Maximum number of episodes across all selected tasks.",
    )
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    p.add_argument(
        "--out_json", type=str, default=None, help="Optional path to dump per-task results JSON."
    )
    p.add_argument(
        "--rollout_dir",
        type=str,
        default=None,
        help="Where to save rollout MP4s. Default: <student_dir>/rollouts/<suite>/",
    )
    p.add_argument(
        "--no_videos", action="store_true", help="Skip MP4 saving (faster, uses less disk)."
    )
    p.add_argument(
        "--min_pixels",
        type=int,
        default=224 * 224,
        help="Min pixels for image preprocessing. MUST match the value used at training (default: 224×224 = 50176).",
    )
    p.add_argument(
        "--max_pixels",
        type=int,
        default=224 * 224,
        help="Max pixels for image preprocessing. MUST match the value used at training (default: 224×224 = 50176).",
    )
    p.add_argument(
        "--max_cot_tokens",
        type=int,
        default=120,
        help="Maximum number of generated reasoning tokens per action chunk.",
    )
    p.add_argument(
        "--print_cot",
        action="store_true",
        help="Print generated Plan and Think text during rollout.",
    )
    p.add_argument(
        "--no_cot_log",
        action="store_true",
        help="Skip writing per-trial CoT transcript JSONL files.",
    )
    p.add_argument(
        "--cot_log_dir",
        type=str,
        default=None,
        help="Override directory for CoT transcript JSONLs. Default: <student_dir>/cot_logs/<suite>/",
    )
    p.add_argument(
        "--gripper_transform",
        type=str,
        default="normalize_only",
        choices=["normalize_invert", "normalize_only", "none"],
        help="Transform the gripper channel before environment execution. Use 'none' for raw output.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    results = eval_suite(args)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
