#!/usr/bin/env python3
"""Execute LIBERO-Plus evaluation tasks from a shared SQLite work queue on one GPU."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import io
import json
import logging
import os
import platform
import random
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import AutoProcessor

LOG = logging.getLogger("libero-plus-dynamic-worker")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def load_module_from_path(name: str, path: Path):
    path = path.resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def is_under(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def connect_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=120.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=120000")
    return conn


def claim_job(conn: sqlite3.Connection, worker_id: str, gpu_id: int) -> Optional[sqlite3.Row]:
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "\n                SELECT * FROM jobs\n                WHERE status='pending'\n                ORDER BY priority ASC, job_id ASC\n                LIMIT 1\n                "
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            now = time.time()
            changed = conn.execute(
                "\n                UPDATE jobs\n                SET status='running', worker_id=?, gpu_id=?, claimed_at=?,\n                    attempts=attempts+1, updated_at=?\n                WHERE job_id=? AND status='pending'\n                ",
                (worker_id, gpu_id, now, now, int(row["job_id"])),
            ).rowcount
            conn.execute("COMMIT")
            if changed == 1:
                return conn.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (int(row["job_id"]),)
                ).fetchone()
        except sqlite3.OperationalError as exc:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            if "locked" not in str(exc).lower():
                raise
            time.sleep(0.2)


def finish_job(
    conn: sqlite3.Connection,
    job_id: int,
    status: str,
    success: Optional[bool],
    result: Dict[str, Any],
    error: Optional[str],
    retry: bool,
) -> None:
    now = time.time()
    if retry:
        new_status = "pending"
        worker_id = None
        gpu_id = None
        claimed_at = None
    else:
        new_status = status
        worker_id = result.get("worker_id")
        gpu_id = result.get("physical_gpu_id")
        claimed_at = result.get("claimed_at")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "\n        UPDATE jobs\n        SET status=?, success=?, result_json=?, last_error=?, finished_at=?,\n            updated_at=?, worker_id=?, gpu_id=?, claimed_at=?\n        WHERE job_id=?\n        ",
        (
            new_status,
            None if success is None else int(bool(success)),
            json.dumps(result, ensure_ascii=False, default=str),
            error,
            None if retry else now,
            now,
            worker_id,
            gpu_id,
            claimed_at,
            int(job_id),
        ),
    )
    conn.execute("COMMIT")


def instantiate_suite(inf: Any, suite_name: str):
    with contextlib.redirect_stdout(io.StringIO()):
        mapping = inf.benchmark.get_benchmark_dict()
        if suite_name not in mapping:
            raise KeyError(f"Unknown suite {suite_name}; available={sorted(mapping)}")
        return mapping[suite_name]()


def load_classification(path: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    output: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for suite, items in raw.items():
        mapping: Dict[int, Dict[str, Any]] = {}
        for idx, item in enumerate(items):
            meta = dict(item)
            raw_id = meta.get("id")
            try:
                one_based_id = int(raw_id)
            except Exception:
                one_based_id = idx + 1
            mapping[one_based_id - 1] = meta
            mapping.setdefault(idx, meta)
        output[str(suite)] = mapping
    return output


def inspect_runtime(inf: Any, libero_plus_root: Path) -> Dict[str, Any]:
    if hasattr(inf, "load_libero_runtime"):
        inf.load_libero_runtime()
    import libero_plus
    import libero_plus.libero
    from libero_plus.libero import benchmark as plus_benchmark
    from libero_plus.libero import envs as plus_envs

    modules = {
        "libero_plus": Path(libero_plus.__file__).resolve(),
        "libero_plus.libero": Path(libero_plus.libero.__file__).resolve(),
        "benchmark": Path(plus_benchmark.__file__).resolve(),
        "envs": Path(plus_envs.__file__).resolve(),
    }
    for name, path in modules.items():
        if not is_under(path, libero_plus_root):
            raise RuntimeError(f"{name} imported from wrong repository: {path}")
    paths = {
        key: str(Path(inf.get_libero_path(key)).resolve())
        for key in ["benchmark_root", "bddl_files", "init_states", "datasets", "assets"]
    }
    for key in ["benchmark_root", "bddl_files", "init_states", "assets"]:
        p = Path(paths[key])
        if not p.exists():
            raise FileNotFoundError(f"Missing LIBERO-Plus {key}: {p}")
        if not is_under(p, libero_plus_root) and key != "assets":
            raise RuntimeError(f"LIBERO-Plus {key} escapes LIBERO-Plus source tree: {p}")
    return {
        "created_at": utc_now(),
        "modules": {k: str(v) for k, v in modules.items()},
        "paths": paths,
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
    }


def load_student(
    inf: Any,
    student_dir: Path,
    processor_dir: Path,
    dtype: torch.dtype,
    attn_impl: str,
    min_pixels: int,
    max_pixels: int,
    model_id_override: Optional[str],
):
    meta = json.loads((student_dir / "action_head_config.json").read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(
        str(processor_dir), trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
    cot_ids = meta.get("cot_token_ids")
    uses_cot = bool(meta.get("uses_cot", False) and cot_ids is not None)
    if uses_cot:
        token_map = {
            "plan_open": inf.COT_PLAN_OPEN,
            "plan_close": inf.COT_PLAN_CLOSE,
            "think_open": inf.COT_THINK_OPEN,
            "think_close": inf.COT_THINK_CLOSE,
        }
        for name, expected_id in cot_ids.items():
            actual_id = processor.tokenizer.convert_tokens_to_ids(token_map[name])
            if int(actual_id) != int(expected_id):
                processor.tokenizer.add_special_tokens(
                    {"additional_special_tokens": inf.COT_SPECIAL_TOKENS}
                )
                break
    model_id = model_id_override or meta["model_id"]
    model = inf.Qwen35ActionPolicy(
        model_id=model_id,
        action_dim=int(meta["action_dim"]),
        proprio_dim=int(meta["proprio_dim"]),
        chunk_size=int(meta["chunk_size"]),
        use_proprio=bool(meta["use_proprio"]),
        dtype=dtype,
        attn_impl=attn_impl,
        cot_token_ids=cot_ids if uses_cot else None,
    )
    if uses_cot:
        target_vocab = int(meta.get("vocab_size", len(processor.tokenizer)))
        if model.vlm.get_input_embeddings().num_embeddings != target_vocab:
            model.vlm.resize_token_embeddings(target_vocab)
    state_path = student_dir / "student_state_dict.pt"
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    except Exception:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        LOG.warning("Missing keys: %d; first=%s", len(missing), missing[:5])
    if unexpected:
        LOG.warning("Unexpected keys: %d; first=%s", len(unexpected), unexpected[:5])
    runtime_meta = dict(meta)
    runtime_meta["model_id_runtime"] = model_id
    return (model, processor, runtime_meta)


def deterministic_episode_seed(base_seed: int, suite: str, task_id: int) -> int:
    suite_code = sum(((i + 1) * ord(ch) for i, ch in enumerate(suite)))
    return int((base_seed * 1000003 + suite_code * 10007 + task_id * 101) % (2**31 - 1))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probe(args: argparse.Namespace, inf: Any) -> int:
    libero_plus_root = Path(args.libero_plus_root).resolve()
    runtime_info = inspect_runtime(inf, libero_plus_root)
    classification = load_classification(Path(args.classification_path).resolve())
    expected = {
        "libero_spatial": 2402,
        "libero_object": 2518,
        "libero_goal": 2591,
        "libero_10": 2519,
    }
    suites_out: Dict[str, Any] = {}
    for suite_name in [x.strip() for x in args.suites.split(",") if x.strip()]:
        suite = instantiate_suite(inf, suite_name)
        n_tasks = int(suite.n_tasks)
        if args.strict_counts and suite_name in expected and (n_tasks != expected[suite_name]):
            raise RuntimeError(
                f"Unexpected {suite_name} task count: {n_tasks}, expected {expected[suite_name]}"
            )
        cls_count = len({id(v) for v in classification.get(suite_name, {}).values()})
        if cls_count not in (0, n_tasks):
            raise RuntimeError(
                f"Classification count mismatch for {suite_name}: {cls_count} vs {n_tasks}"
            )
        task = suite.get_task(0)
        states = suite.get_task_init_states(0)
        bddl = Path(inf.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = inf.OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        try:
            env.seed(args.seed)
            env.reset()
            env.set_init_state(states[0])
            obs = None
            for _ in range(2):
                obs, _, _, _ = env.step(inf.libero_dummy_action())
            if obs is None or "agentview_image" not in obs or "robot0_eye_in_hand_image" not in obs:
                raise RuntimeError("Render probe returned incomplete observation")
        finally:
            env.close()
        suites_out[suite_name] = {
            "n_tasks": n_tasks,
            "classification_items": cls_count,
            "sample_task": task.language,
            "sample_bddl": str(bddl.resolve()),
            "init_states_shape": list(states.shape),
            "render_ok": True,
        }
    result = {"runtime": runtime_info, "suites": suites_out, "created_at": utc_now()}
    atomic_json(Path(args.probe_output), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def evaluate(args: argparse.Namespace, inf: Any) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in worker")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    libero_plus_root = Path(args.libero_plus_root).resolve()
    runtime_info = inspect_runtime(inf, libero_plus_root)
    worker_dir = Path(args.worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        worker_dir / "worker_manifest.json",
        {
            "worker_id": args.worker_id,
            "physical_gpu_id": args.physical_gpu_id,
            "args": vars(args),
            "runtime": runtime_info,
        },
    )
    LOG.info(
        "Loading model once on physical GPU %s (visible cuda:0=%s)",
        args.physical_gpu_id,
        torch.cuda.get_device_name(0),
    )
    model, processor, model_meta = load_student(
        inf=inf,
        student_dir=Path(args.student_dir).resolve(),
        processor_dir=Path(args.processor_dir).resolve(),
        dtype=dtype,
        attn_impl=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        model_id_override=args.model_id_override,
    )
    model.to(device).eval()
    classification = load_classification(Path(args.classification_path).resolve())
    suites: Dict[str, Any] = {}
    conn = connect_db(Path(args.db_path).resolve())
    processed = successes = errors = 0
    worker_started = time.time()
    try:
        while True:
            job = claim_job(conn, args.worker_id, args.physical_gpu_id)
            if job is None:
                break
            job_id = int(job["job_id"])
            suite_name = str(job["suite"])
            task_id = int(job["task_id"])
            attempts = int(job["attempts"])
            claimed_at = float(job["claimed_at"])
            if suite_name not in suites:
                suites[suite_name] = instantiate_suite(inf, suite_name)
            suite = suites[suite_name]
            result: Dict[str, Any] = {
                "record_type": "episode",
                "status": "running",
                "job_id": job_id,
                "seed": args.seed,
                "suite": suite_name,
                "task_id": task_id,
                "classification_id": task_id + 1,
                "attempt": attempts,
                "worker_id": args.worker_id,
                "physical_gpu_id": args.physical_gpu_id,
                "claimed_at": claimed_at,
                "started_at": utc_now(),
                "checkpoint": str(Path(args.student_dir).resolve()),
                "model_id_runtime": model_meta.get("model_id_runtime"),
            }
            task_meta = dict(classification.get(suite_name, {}).get(task_id, {}))
            for key, value in task_meta.items():
                result[f"classification_{key}"] = value
            result["category"] = task_meta.get("category")
            result["difficulty_level"] = task_meta.get("difficulty_level")
            result["classification_name"] = task_meta.get("name")
            t0 = time.time()
            env = None
            try:
                task = suite.get_task(task_id)
                result.update(
                    {
                        "task_name": task.name,
                        "task_description": task.language,
                        "problem_folder": task.problem_folder,
                        "bddl_file_name": task.bddl_file,
                    }
                )
                bddl = (
                    Path(inf.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                )
                result["bddl_file"] = str(bddl.resolve())
                init_states = suite.get_task_init_states(task_id)
                init_index = 0 if args.init_state_mode == "first" else args.seed % len(init_states)
                result["init_state_index"] = int(init_index)
                result["n_init_states"] = int(len(init_states))
                episode_seed = deterministic_episode_seed(args.seed, suite_name, task_id)
                result["episode_seed"] = episode_seed
                set_all_seeds(episode_seed)
                env = inf.OffScreenRenderEnv(
                    bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
                )
                env.seed(episode_seed)
                success = inf.rollout_episode(
                    model=model,
                    processor=processor,
                    meta=model_meta,
                    env=env,
                    task_init_state=init_states[init_index],
                    task_desc=task.language,
                    max_steps=args.max_steps,
                    device=device,
                    dtype=dtype,
                    video_path=None,
                    gripper_transform=args.gripper_transform,
                    cot_log_path=None,
                    print_cot=False,
                    max_cot_tokens=args.max_cot_tokens,
                )
                elapsed = time.time() - t0
                result.update(
                    {
                        "status": "ok",
                        "success": bool(success),
                        "elapsed_s": elapsed,
                        "finished_at": utc_now(),
                    }
                )
                finish_job(conn, job_id, "done", bool(success), result, None, retry=False)
                processed += 1
                successes += int(bool(success))
                LOG.info(
                    "%s task %d/%d -> %s [%.1fs] category=%s difficulty=%s",
                    suite_name,
                    task_id + 1,
                    suite.n_tasks,
                    "SUCCESS" if success else "FAIL",
                    elapsed,
                    result.get("category"),
                    result.get("difficulty_level"),
                )
            except Exception as exc:
                elapsed = time.time() - t0
                result.update(
                    {
                        "status": "error",
                        "success": None,
                        "elapsed_s": elapsed,
                        "finished_at": utc_now(),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                retry = attempts <= args.max_retries
                finish_job(conn, job_id, "error", None, result, repr(exc), retry=retry)
                errors += 1
                LOG.exception(
                    "Task failed: suite=%s task_id=%d attempt=%d retry=%s",
                    suite_name,
                    task_id,
                    attempts,
                    retry,
                )
                if args.fail_fast:
                    raise
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                gc.collect()
                torch.cuda.empty_cache()
            if processed % max(1, args.summary_every) == 0:
                atomic_json(
                    worker_dir / "worker_summary.json",
                    {
                        "updated_at": utc_now(),
                        "worker_id": args.worker_id,
                        "physical_gpu_id": args.physical_gpu_id,
                        "processed": processed,
                        "successes": successes,
                        "errors": errors,
                        "elapsed_s": time.time() - worker_started,
                    },
                )
    finally:
        conn.close()
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        worker_dir / "worker_summary.json",
        {
            "updated_at": utc_now(),
            "worker_id": args.worker_id,
            "physical_gpu_id": args.physical_gpu_id,
            "processed": processed,
            "successes": successes,
            "errors": errors,
            "elapsed_s": time.time() - worker_started,
            "finished": True,
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--libero_plus_root", required=True)
    p.add_argument("--infer_script", required=True)
    p.add_argument("--classification_path", required=True)
    p.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--strict_counts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--probe_only", action="store_true")
    p.add_argument("--probe_output")
    p.add_argument("--db_path")
    p.add_argument("--student_dir")
    p.add_argument("--processor_dir")
    p.add_argument("--model_id_override")
    p.add_argument("--worker_id", default="worker")
    p.add_argument("--physical_gpu_id", type=int, default=0)
    p.add_argument("--worker_dir")
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument(
        "--attn_implementation",
        choices=["sdpa", "eager", "flash_attention_2"],
        default="flash_attention_2",
    )
    p.add_argument("--min_pixels", type=int, default=224 * 224)
    p.add_argument("--max_pixels", type=int, default=224 * 224)
    p.add_argument("--max_cot_tokens", type=int, default=120)
    p.add_argument(
        "--gripper_transform",
        choices=["normalize_invert", "normalize_only", "none"],
        default="normalize_only",
    )
    p.add_argument("--init_state_mode", choices=["first", "seeded"], default="first")
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--summary_every", type=int, default=10)
    p.add_argument("--fail_fast", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    inf = load_module_from_path(
        "cotiny_libero_plus_infer_runtime", Path(args.infer_script).resolve()
    )
    if args.probe_only:
        return probe(args, inf)
    for name in ["db_path", "student_dir", "processor_dir", "worker_dir"]:
        if getattr(args, name) is None:
            raise ValueError(f"--{name} is required")
    return evaluate(args, inf)


if __name__ == "__main__":
    raise SystemExit(main())
