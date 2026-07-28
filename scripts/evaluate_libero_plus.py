#!/usr/bin/env python3
"""Schedule LIBERO-Plus evaluation tasks dynamically across available GPUs and aggregate detailed results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXPECTED_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
    "Object Layout",
    "Layout",
]
PLACEHOLDERS = ("REPLACE_WITH_CHECKPOINT",)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def processor_ready(path: Path) -> bool:
    tokenizer = any(
        (
            (path / name).exists()
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        )
    )
    processor = any(
        (
            (path / name).exists()
            for name in (
                "processor_config.json",
                "preprocessor_config.json",
                "image_processor_config.json",
            )
        )
    )
    return tokenizer and processor


def checkpoint_step(path: Path) -> int:
    state = read_json(path / "trainer_state.json")
    try:
        if state.get("global_step") is not None:
            return int(state["global_step"])
    except Exception:
        pass
    m = re.fullmatch("checkpoint-(\\d+)", path.name)
    if m:
        return int(m.group(1))
    child_steps = []
    for child in path.glob("checkpoint-*"):
        m = re.fullmatch("checkpoint-(\\d+)", child.name)
        if m and (child / "student_state_dict.pt").exists():
            child_steps.append(int(m.group(1)))
    return max(child_steps) if child_steps else 0


def find_processor_dir(path: Path, roots: Sequence[Path]) -> Optional[Path]:
    current = path.resolve()
    roots = [x.resolve() for x in roots]
    while True:
        if processor_ready(current):
            return current
        if current.parent == current or current in roots:
            return None
        current = current.parent


def discover_checkpoint(
    explicit: Optional[str], roots: Sequence[Path]
) -> Tuple[Path, Path, Dict[str, Any]]:
    roots = [x.expanduser().resolve() for x in roots if x.expanduser().exists()]
    if not roots:
        raise FileNotFoundError("No checkpoint search roots exist")
    candidates: List[Dict[str, Any]] = []
    explicit_valid = explicit and (not any((mark in explicit for mark in PLACEHOLDERS)))
    paths = set()
    if explicit_valid:
        paths.add(Path(explicit).expanduser().resolve())
    for root in roots:
        for config in root.rglob("action_head_config.json"):
            paths.add(config.parent.resolve())
    for path in sorted(paths):
        config = path / "action_head_config.json"
        state = path / "student_state_dict.pt"
        if not config.exists() or not state.exists() or state.stat().st_size <= 0:
            continue
        meta = read_json(config)
        reasons = []
        if int(meta.get("num_images", -1)) != 8:
            reasons.append(f"num_images={meta.get('num_images')}")
        if int(meta.get("num_wrist_images", 0)) != 8:
            reasons.append(f"num_wrist_images={meta.get('num_wrist_images')}")
        if not bool(meta.get("uses_cot", False)):
            reasons.append("uses_cot=false")
        processor = find_processor_dir(path, roots)
        if processor is None:
            reasons.append("processor/tokenizer not found")
        step = checkpoint_step(path)
        final_export = not bool(re.fullmatch("checkpoint-\\d+", path.name))
        candidates.append(
            {
                "path": str(path),
                "processor_dir": str(processor) if processor else None,
                "global_step": step,
                "final_export": final_export,
                "state_size_gib": state.stat().st_size / 1024**3,
                "model_id": meta.get("model_id"),
                "valid": not reasons,
                "reasons": reasons,
                "mtime": max(config.stat().st_mtime, state.stat().st_mtime),
            }
        )
    candidates.sort(
        key=lambda r: (int(r["global_step"]), int(r["final_export"]), float(r["mtime"])),
        reverse=True,
    )
    valid = [x for x in candidates if x["valid"]]
    if explicit_valid:
        explicit_path = str(Path(explicit).expanduser().resolve())
        explicit_rows = [x for x in valid if x["path"] == explicit_path]
        if explicit_rows:
            selected = explicit_rows[0]
        elif valid:
            print(
                "WARNING: explicit checkpoint invalid; using latest valid checkpoint",
                file=sys.stderr,
            )
            selected = valid[0]
        else:
            raise FileNotFoundError("No valid checkpoint found")
    elif valid:
        selected = valid[0]
    else:
        raise FileNotFoundError("No valid dual-view CoTinyVLA checkpoint found")
    report = {
        "created_at": now_iso(),
        "search_roots": [str(x) for x in roots],
        "explicit": explicit,
        "selected": selected,
        "candidates": candidates,
    }
    return (Path(selected["path"]), Path(selected["processor_dir"]), report)


def print_checkpoint_report(report: Dict[str, Any]) -> None:
    print("\n===== CoTinyVLA checkpoint candidates =====")
    print(f"{'rank':>4} {'step':>8} {'final':>6} {'valid':>6} {'GiB':>7}  path")
    print("-" * 120)
    for idx, row in enumerate(report["candidates"][:20], 1):
        print(
            f"{idx:>4} {int(row['global_step']):>8} {str(row['final_export']):>6} {str(row['valid']):>6} {float(row['state_size_gib']):>7.2f}  {row['path']}"
        )
        if row["reasons"]:
            print("      rejected: " + "; ".join(row["reasons"]))
    print("-" * 120)
    print("SELECTED:", report["selected"]["path"])
    print("PROCESSOR:", report["selected"]["processor_dir"])
    print()


def connect_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=120.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=120000")
    return conn


def initialize_queue(
    db_path: Path,
    suites: List[str],
    counts: Dict[str, int],
    task_limit: Optional[int],
    reset_running: bool,
) -> Dict[str, int]:
    conn = connect_db(db_path)
    conn.execute(
        "\n        CREATE TABLE IF NOT EXISTS jobs (\n            job_id INTEGER PRIMARY KEY AUTOINCREMENT,\n            suite TEXT NOT NULL,\n            task_id INTEGER NOT NULL,\n            priority INTEGER NOT NULL,\n            status TEXT NOT NULL DEFAULT 'pending',\n            attempts INTEGER NOT NULL DEFAULT 0,\n            worker_id TEXT,\n            gpu_id INTEGER,\n            claimed_at REAL,\n            finished_at REAL,\n            updated_at REAL,\n            success INTEGER,\n            result_json TEXT,\n            last_error TEXT,\n            UNIQUE(suite, task_id)\n        )\n        "
    )
    if reset_running:
        conn.execute(
            "\n            UPDATE jobs SET status='pending', worker_id=NULL, gpu_id=NULL,\n                claimed_at=NULL, updated_at=? WHERE status='running'\n            ",
            (time.time(),),
        )
    priority = 0
    for round_id in range(max(counts.values())):
        for suite in suites:
            n = counts[suite]
            if task_limit is not None:
                n = min(n, task_limit)
            if round_id >= n:
                continue
            conn.execute(
                "\n                INSERT OR IGNORE INTO jobs(suite, task_id, priority, status, updated_at)\n                VALUES(?, ?, ?, 'pending', ?)\n                ",
                (suite, round_id, priority, time.time()),
            )
            priority += 1
    stats = queue_stats(conn)
    conn.close()
    return stats


def queue_stats(conn: sqlite3.Connection) -> Dict[str, int]:
    counts = {
        row["status"]: int(row["n"])
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
    }
    for key in ["pending", "running", "done", "error"]:
        counts.setdefault(key, 0)
    counts["total"] = sum((counts[k] for k in ["pending", "running", "done", "error"]))
    counts["successes"] = int(
        conn.execute("SELECT COUNT(*) FROM jobs WHERE status='done' AND success=1").fetchone()[0]
    )
    return counts


def parse_nvidia_smi() -> Dict[int, Dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    states: Dict[int, Dict[str, float]] = {}
    for line in proc.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue

        def number(value: str, default: float = 0.0) -> float:
            value = value.strip()
            if value.upper() in {"N/A", "NA", "[N/A]", "-"}:
                return default
            return float(value)

        idx = number(parts[0])
        used = number(parts[1])
        total = number(parts[2])
        util = number(parts[3])
        states[int(idx)] = {
            "memory_used_mb": used,
            "memory_total_mb": total,
            "utilization_pct": util,
            "memory_free_mb": total - used,
        }
    return states


def gpu_is_idle(
    state: Dict[str, float], max_used_mb: float, max_util_pct: float, min_free_mb: float
) -> bool:
    return (
        state["memory_used_mb"] <= max_used_mb
        and state["utilization_pct"] <= max_util_pct
        and (state["memory_free_mb"] >= min_free_mb)
    )


def build_worker_env(
    libero_plus_root: Path,
    config_dir: Path,
    project_root: Path,
    gpu: int,
    egl_mode: str,
    offline: bool,
) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(libero_plus_root.resolve()), str(project_root.resolve())]
    )
    env["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())
    env["MUJOCO_GL"] = "egl"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if egl_mode == "physical":
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    elif egl_mode == "zero":
        env["MUJOCO_EGL_DEVICE_ID"] = "0"
    else:
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def write_plus_config(
    libero_plus_root: Path, config_dir: Path, assets_override: Optional[Path]
) -> Dict[str, str]:
    inner = libero_plus_root / "libero_plus" / "libero"
    if assets_override is not None:
        assets = assets_override.expanduser().resolve()
    elif (inner / "assets").exists():
        assets = (inner / "assets").resolve()
    else:
        assets = (inner / "assets").resolve()
    paths = {
        "benchmark_root": str(inner.resolve()),
        "bddl_files": str((inner / "bddl_files").resolve()),
        "init_states": str((inner / "init_files").resolve()),
        "datasets": str((libero_plus_root / "libero_plus" / "datasets").resolve()),
        "assets": str(assets),
    }
    Path(paths["datasets"]).mkdir(parents=True, exist_ok=True)
    for key in ["benchmark_root", "bddl_files", "init_states", "assets"]:
        if not Path(paths[key]).exists():
            raise FileNotFoundError(f"Missing {key}: {paths[key]}")
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        "\n".join((f"{k}: {json.dumps(v)}" for k, v in paths.items())) + "\n", encoding="utf-8"
    )
    return paths


def run_probe(
    args: argparse.Namespace,
    gpu: int,
    student_dir: Path,
    processor_dir: Path,
    classification_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output = output_dir / "preflight.json"
    log = output_dir / "preflight.log"
    env = build_worker_env(
        args.libero_plus_root,
        args.config_dir,
        args.project_root,
        gpu,
        args.egl_device_mode,
        args.offline,
    )
    cmd = [
        sys.executable,
        str(args.worker_script),
        "--libero_plus_root",
        str(args.libero_plus_root),
        "--infer_script",
        str(args.infer_script),
        "--classification_path",
        str(classification_path),
        "--suites",
        ",".join(args.suites),
        "--seed",
        str(args.seed),
        "--probe_only",
        "--probe_output",
        str(output),
    ]
    if not args.strict_counts:
        cmd.append("--no-strict_counts")
    with log.open("w", buffering=1) as handle:
        rc = subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-150:])
        raise RuntimeError(f"LIBERO-Plus preflight failed:\n{tail}")
    return json.loads(output.read_text(encoding="utf-8"))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (100 * (center - half), 100 * (center + half))


def aggregate_group(rows: List[Dict[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple((row.get(field) for field in fields))].append(row)
    output = []
    for key, group in groups.items():
        n = len(group)
        s = sum((bool(r.get("success")) for r in group))
        lo, hi = wilson(s, n)
        entry = {field: value for field, value in zip(fields, key)}
        entry.update(
            {
                "episodes": n,
                "successes": s,
                "success_rate_pct": 100 * s / n if n else None,
                "wilson95_low_pct": lo,
                "wilson95_high_pct": hi,
                "mean_elapsed_s": (
                    sum((float(r.get("elapsed_s", 0)) for r in group)) / n if n else None
                ),
            }
        )
        output.append(entry)
    return output


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_results(conn: sqlite3.Connection) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    done = []
    errors = []
    for row in conn.execute(
        "SELECT * FROM jobs WHERE status IN ('done','error') ORDER BY suite, task_id"
    ):
        payload = {}
        if row["result_json"]:
            try:
                payload = json.loads(row["result_json"])
            except Exception:
                payload = {}
        payload.setdefault("suite", row["suite"])
        payload.setdefault("task_id", int(row["task_id"]))
        payload.setdefault("success", None if row["success"] is None else bool(row["success"]))
        payload["queue_status"] = row["status"]
        payload["queue_attempts"] = int(row["attempts"])
        payload["queue_last_error"] = row["last_error"]
        (done if row["status"] == "done" else errors).append(payload)
    return (done, errors)


def render_markdown_table(title: str, rows: List[Dict[str, Any]], label_field: str) -> str:
    lines = [
        f"## {title}",
        "",
        "| Group | Successes | Tasks | SR | Wilson 95% CI | Mean time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = row.get(label_field)
        lines.append(
            f"| {label} | {row['successes']} | {row['episodes']} | {row['success_rate_pct']:.2f}% | [{row['wilson95_low_pct']:.2f}, {row['wilson95_high_pct']:.2f}]% | {row['mean_elapsed_s']:.1f}s |"
        )
    return "\n".join(lines)


def aggregate_outputs(db_path: Path, output_dir: Path, suites: List[str]) -> Dict[str, Any]:
    conn = connect_db(db_path)
    valid, errors = parse_results(conn)
    stats = queue_stats(conn)
    conn.close()
    write_csv(output_dir / "episodes_detailed.csv", valid)
    with (output_dir / "episodes_detailed.jsonl").open("w", encoding="utf-8") as handle:
        for row in valid:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    write_csv(output_dir / "errors.csv", errors)
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in errors:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    suite_rows = aggregate_group(valid, ["suite"])
    suite_rows.sort(key=lambda r: suites.index(r["suite"]) if r["suite"] in suites else 999)
    category_rows = aggregate_group(valid, ["category"])
    category_rows.sort(
        key=lambda r: (
            CATEGORY_ORDER.index(r["category"]) if r.get("category") in CATEGORY_ORDER else 999
        )
    )
    difficulty_rows = aggregate_group(valid, ["difficulty_level"])
    difficulty_rows.sort(key=lambda r: str(r.get("difficulty_level")))
    category_difficulty_rows = aggregate_group(valid, ["category", "difficulty_level"])
    category_difficulty_rows.sort(
        key=lambda r: (str(r.get("category")), str(r.get("difficulty_level")))
    )
    suite_category_rows = aggregate_group(valid, ["suite", "category"])
    suite_category_rows.sort(
        key=lambda r: (
            suites.index(r["suite"]) if r["suite"] in suites else 999,
            str(r.get("category")),
        )
    )
    suite_difficulty_rows = aggregate_group(valid, ["suite", "difficulty_level"])
    suite_difficulty_rows.sort(
        key=lambda r: (
            suites.index(r["suite"]) if r["suite"] in suites else 999,
            str(r.get("difficulty_level")),
        )
    )
    base_task_rows = aggregate_group(valid, ["suite", "classification_name"])
    base_task_rows.sort(
        key=lambda r: (
            suites.index(r["suite"]) if r["suite"] in suites else 999,
            str(r.get("classification_name")),
        )
    )
    outputs = {
        "success_by_suite.csv": suite_rows,
        "success_by_perturbation.csv": category_rows,
        "success_by_difficulty.csv": difficulty_rows,
        "success_by_perturbation_and_difficulty.csv": category_difficulty_rows,
        "success_by_suite_and_perturbation.csv": suite_category_rows,
        "success_by_suite_and_difficulty.csv": suite_difficulty_rows,
        "success_by_base_task.csv": base_task_rows,
    }
    for name, rows in outputs.items():
        write_csv(output_dir / name, rows)
    successes = sum((bool(r.get("success")) for r in valid))
    low, high = wilson(successes, len(valid))
    summary = {
        "created_at": now_iso(),
        "queue": stats,
        "valid_tasks": len(valid),
        "error_tasks": len(errors),
        "successes": successes,
        "success_rate_pct": 100 * successes / len(valid) if valid else None,
        "wilson95_low_pct": low if valid else None,
        "wilson95_high_pct": high if valid else None,
        "complete": stats["pending"] == 0 and stats["running"] == 0 and (stats["error"] == 0),
        "by_suite": suite_rows,
        "by_perturbation": category_rows,
        "by_difficulty": difficulty_rows,
    }
    atomic_json(output_dir / "summary.json", summary)
    md = [
        "# LIBERO-Plus single-seed evaluation",
        "",
        f"- Valid tasks: {len(valid)}",
        f"- Errors: {len(errors)}",
        (
            f"- Success: {successes}/{len(valid)} = {summary['success_rate_pct']:.2f}%"
            if valid
            else "- Success: n/a"
        ),
        f"- Wilson 95% CI: [{low:.2f}, {high:.2f}]%" if valid else "- Wilson 95% CI: n/a",
        f"- Complete: {summary['complete']}",
        "",
        render_markdown_table("By suite", suite_rows, "suite"),
        "",
        render_markdown_table("By perturbation", category_rows, "category"),
        "",
        render_markdown_table("By difficulty", difficulty_rows, "difficulty_level"),
    ]
    (output_dir / "success_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Single-seed LIBERO-Plus success rate by suite.}",
        "\\label{tab:libero-plus-suite}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Suite & Successes & Tasks & SR (\\%) \\\\",
        "\\midrule",
    ]
    for row in suite_rows:
        tex.append(
            f"{row['suite']} & {row['successes']} & {row['episodes']} & {row['success_rate_pct']:.2f} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (output_dir / "success_by_suite.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return summary


def live_progress(db_path: Path, started: float) -> Dict[str, Any]:
    conn = connect_db(db_path)
    stats = queue_stats(conn)
    suite_rows = []
    for row in conn.execute(
        "\n        SELECT suite,\n               SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done_n,\n               SUM(CASE WHEN status='done' AND success=1 THEN 1 ELSE 0 END) AS succ_n,\n               SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS err_n\n        FROM jobs GROUP BY suite ORDER BY suite\n        "
    ):
        suite_rows.append(dict(row))
    conn.close()
    elapsed = time.time() - started
    completed = stats["done"] + stats["error"]
    rate = completed / elapsed if elapsed > 0 else 0
    eta = stats["pending"] / rate if rate > 0 else None
    return {
        "stats": stats,
        "suites": suite_rows,
        "elapsed_s": elapsed,
        "rate_tasks_per_s": rate,
        "eta_s": eta,
    }


def print_progress(progress: Dict[str, Any], active_gpus: Sequence[int]) -> None:
    s = progress["stats"]
    eta = progress["eta_s"]
    eta_text = f"{eta / 3600:.2f}h" if eta is not None else "n/a"
    sr = 100 * s["successes"] / s["done"] if s["done"] else 0
    print(
        f"[{now_iso()}] done={s['done']}/{s['total']} pending={s['pending']} running={s['running']} errors={s['error']} SR={sr:.2f}% active_gpus={list(active_gpus)} ETA={eta_text}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    script_dir = Path(__file__).resolve().parent
    p.add_argument("--libero_plus_root", type=Path, required=True)
    p.add_argument("--config_dir", type=Path, default=None)
    p.add_argument("--assets_dir", type=Path, default=None)
    p.add_argument("--project_root", type=Path, default=script_dir)
    p.add_argument("--infer_script", type=Path, default=script_dir / "infer_libero_plus.py")
    p.add_argument(
        "--worker_script", type=Path, default=script_dir / "evaluate_libero_plus_worker.py"
    )
    p.add_argument("--classification_path", type=Path, default=None)
    p.add_argument("--checkpoint_search_root", action="append", type=Path, default=None)
    p.add_argument("--student_dir", type=str, default=None)
    p.add_argument("--model_id_override", default=None)
    p.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--task_limit_per_suite", type=int, default=None)
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
    p.add_argument("--egl_device_mode", choices=["physical", "zero", "unset"], default="physical")
    p.add_argument("--max_gpu_used_memory_mb", type=float, default=1500)
    p.add_argument("--max_gpu_utilization_pct", type=float, default=10)
    p.add_argument("--min_gpu_free_memory_mb", type=float, default=8000)
    p.add_argument("--poll_interval_sec", type=float, default=10)
    p.add_argument("--progress_interval_sec", type=float, default=60)
    p.add_argument("--launch_stagger_sec", type=float, default=2)
    p.add_argument("--gpu_wait_timeout_sec", type=float, default=0)
    p.add_argument("--strict_counts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--fail_fast", action="store_true")
    p.add_argument("--aggregate_only", action="store_true")
    p.add_argument("--list_checkpoints", action="store_true")
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()
    args.libero_plus_root = args.libero_plus_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir = (
        args.config_dir.expanduser().resolve()
        if args.config_dir
        else args.output_dir / "libero_config"
    )
    args.assets_dir = args.assets_dir.expanduser().resolve() if args.assets_dir else None
    args.project_root = args.project_root.expanduser().resolve()
    args.infer_script = args.infer_script.expanduser().resolve()
    args.worker_script = args.worker_script.expanduser().resolve()
    args.suites = parse_csv(args.suites)
    args.gpus = parse_int_csv(args.gpus)
    args.classification_path = (
        args.classification_path.expanduser().resolve()
        if args.classification_path
        else args.libero_plus_root
        / "libero_plus"
        / "libero"
        / "benchmark"
        / "task_classification.json"
    )
    for path in [
        args.libero_plus_root,
        args.infer_script,
        args.worker_script,
        args.classification_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    db_path = args.output_dir / "jobs.sqlite3"
    if args.aggregate_only:
        summary = aggregate_outputs(db_path, args.output_dir, args.suites)
        print((args.output_dir / "success_tables.md").read_text())
        return 0
    roots = args.checkpoint_search_root if args.checkpoint_search_root else [args.project_root]
    student_dir, processor_dir, checkpoint_report = discover_checkpoint(args.student_dir, roots)
    print_checkpoint_report(checkpoint_report)
    atomic_json(args.output_dir / "checkpoint_selection.json", checkpoint_report)
    if args.list_checkpoints:
        return 0
    config_paths = write_plus_config(args.libero_plus_root, args.config_dir, args.assets_dir)
    atomic_json(args.output_dir / "path_config.json", config_paths)
    wait_started = time.time()
    probe_gpu = None
    while probe_gpu is None:
        states = parse_nvidia_smi()
        for gpu in args.gpus:
            state = states.get(gpu)
            if state and gpu_is_idle(
                state,
                args.max_gpu_used_memory_mb,
                args.max_gpu_utilization_pct,
                args.min_gpu_free_memory_mb,
            ):
                probe_gpu = gpu
                break
        if probe_gpu is None:
            if (
                args.gpu_wait_timeout_sec > 0
                and time.time() - wait_started > args.gpu_wait_timeout_sec
            ):
                raise TimeoutError("No idle GPU became available for preflight")
            print("No idle GPU available for preflight; waiting...", flush=True)
            time.sleep(args.poll_interval_sec)
    probe = run_probe(
        args, probe_gpu, student_dir, processor_dir, args.classification_path, args.output_dir
    )
    counts = {suite: int(probe["suites"][suite]["n_tasks"]) for suite in args.suites}
    queue_initial = initialize_queue(
        db_path, args.suites, counts, args.task_limit_per_suite, reset_running=True
    )
    atomic_json(
        args.output_dir / "run_manifest.json",
        {
            "created_at": now_iso(),
            "seed": args.seed,
            "suites": args.suites,
            "candidate_gpus": args.gpus,
            "student_dir": str(student_dir),
            "processor_dir": str(processor_dir),
            "libero_plus_root": str(args.libero_plus_root),
            "classification_path": str(args.classification_path),
            "config_paths": config_paths,
            "queue_initial": queue_initial,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    workers_dir = args.output_dir / "workers"
    logs_dir = args.output_dir / "logs"
    workers_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    active: Dict[int, Dict[str, Any]] = {}
    worker_counter = 0
    started = time.time()
    last_progress = 0.0
    no_gpu_since: Optional[float] = None

    def launch_worker(gpu: int) -> None:
        nonlocal worker_counter
        worker_id = f"worker_{worker_counter:03d}_gpu{gpu}"
        worker_counter += 1
        worker_dir = workers_dir / worker_id
        worker_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{worker_id}.log"
        env = build_worker_env(
            args.libero_plus_root,
            args.config_dir,
            args.project_root,
            gpu,
            args.egl_device_mode,
            args.offline,
        )
        cmd = [
            sys.executable,
            str(args.worker_script),
            "--libero_plus_root",
            str(args.libero_plus_root),
            "--infer_script",
            str(args.infer_script),
            "--classification_path",
            str(args.classification_path),
            "--suites",
            ",".join(args.suites),
            "--seed",
            str(args.seed),
            "--db_path",
            str(db_path),
            "--student_dir",
            str(student_dir),
            "--processor_dir",
            str(processor_dir),
            "--worker_id",
            worker_id,
            "--physical_gpu_id",
            str(gpu),
            "--worker_dir",
            str(worker_dir),
            "--max_steps",
            str(args.max_steps),
            "--attn_implementation",
            args.attn_implementation,
            "--min_pixels",
            str(args.min_pixels),
            "--max_pixels",
            str(args.max_pixels),
            "--max_cot_tokens",
            str(args.max_cot_tokens),
            "--gripper_transform",
            args.gripper_transform,
            "--init_state_mode",
            args.init_state_mode,
            "--max_retries",
            str(args.max_retries),
        ]
        if args.model_id_override:
            cmd += ["--model_id_override", args.model_id_override]
        if args.fail_fast:
            cmd.append("--fail_fast")
        log_handle = log_path.open("a", buffering=1)
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        active[gpu] = {"proc": proc, "handle": log_handle, "worker_id": worker_id, "log": log_path}
        print(f"Started {worker_id} on GPU {gpu}, pid={proc.pid}", flush=True)

    try:
        while True:
            for gpu in list(active):
                item = active[gpu]
                rc = item["proc"].poll()
                if rc is None:
                    continue
                item["handle"].close()
                print(f"{item['worker_id']} on GPU {gpu} exited rc={rc}", flush=True)
                if rc != 0:
                    conn = connect_db(db_path)
                    conn.execute(
                        "\n                        UPDATE jobs SET status='pending', worker_id=NULL, gpu_id=NULL,\n                            claimed_at=NULL, updated_at=?\n                        WHERE status='running' AND worker_id=?\n                        ",
                        (time.time(), item["worker_id"]),
                    )
                    conn.close()
                del active[gpu]
            conn = connect_db(db_path)
            stats = queue_stats(conn)
            conn.close()
            if stats["pending"] == 0 and stats["running"] == 0:
                break
            states = parse_nvidia_smi()
            launched_any = False
            for gpu in args.gpus:
                if gpu in active or stats["pending"] <= 0:
                    continue
                state = states.get(gpu)
                if state and gpu_is_idle(
                    state,
                    args.max_gpu_used_memory_mb,
                    args.max_gpu_utilization_pct,
                    args.min_gpu_free_memory_mb,
                ):
                    launch_worker(gpu)
                    launched_any = True
                    stats["pending"] -= 1
                    time.sleep(args.launch_stagger_sec)
            if not active and (not launched_any) and (stats["pending"] > 0):
                if no_gpu_since is None:
                    no_gpu_since = time.time()
                if (
                    args.gpu_wait_timeout_sec > 0
                    and time.time() - no_gpu_since > args.gpu_wait_timeout_sec
                ):
                    raise TimeoutError("Pending tasks remain but no idle GPU became available")
            else:
                no_gpu_since = None
            if time.time() - last_progress >= args.progress_interval_sec:
                progress = live_progress(db_path, started)
                print_progress(progress, sorted(active))
                atomic_json(args.output_dir / "progress.json", progress)
                aggregate_outputs(db_path, args.output_dir, args.suites)
                last_progress = time.time()
            time.sleep(args.poll_interval_sec)
    except KeyboardInterrupt:
        print("Interrupted; terminating worker process groups...", file=sys.stderr)
        for item in active.values():
            try:
                os.killpg(item["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            item["handle"].close()
        raise
    summary = aggregate_outputs(db_path, args.output_dir, args.suites)
    print("\nFinished. Results:", args.output_dir)
    print((args.output_dir / "success_tables.md").read_text())
    return 0 if summary["error_tasks"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
