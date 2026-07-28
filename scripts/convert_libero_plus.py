#!/usr/bin/env python3
"""Convert a LeRobot LIBERO dataset into temporal action-chunk JSONL with optional wrist images."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(quat[:3])
    if den < 1e-09:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(den, quat[3])
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi
    return quat[:3] / den * angle


def to_256(img_arr: np.ndarray) -> np.ndarray:
    if img_arr.shape[0] != 256 or img_arr.shape[1] != 256:
        img_arr = np.array(
            Image.fromarray(img_arr).resize((256, 256), Image.BILINEAR), dtype=np.uint8
        )
    return np.ascontiguousarray(img_arr)


def pad_history(frames: list, n: int) -> list:
    if len(frames) >= n:
        return frames[-n:]
    if not frames:
        raise ValueError("empty history")
    return [frames[0]] * (n - len(frames)) + list(frames)


def save_image(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def find_dataset_root(root: Path) -> Path:
    if (root / "meta" / "info.json").exists():
        return root
    candidates = list(root.rglob("meta/info.json"))
    if not candidates:
        sys.exit(f"No meta/info.json under {root}")
    new_root = candidates[0].parent.parent
    print(f"Detected dataset root: {new_root}")
    return new_root


def load_tasks(root: Path) -> dict[int, str]:
    jsonl = root / "meta" / "tasks.jsonl"
    if jsonl.exists():
        out = {}
        with open(jsonl) as f:
            for line in f:
                d = json.loads(line)
                out[int(d["task_index"])] = d["task"]
        return out
    parquets = sorted((root / "meta").rglob("tasks*.parquet"))
    if parquets:
        df = pd.read_parquet(parquets[0])
        return {int(k): v for k, v in zip(df["task_index"], df["task"])}
    sys.exit("No meta/tasks.jsonl or tasks.parquet")


def load_data_iter(root: Path):
    for path in sorted((root / "data").rglob("*.parquet")):
        yield (path, pd.read_parquet(path))


class ImageProvider:

    def __init__(self, root: Path, info: dict, include_wrist: bool = True):
        self.root = root
        self.info = info
        self.video_template = info.get("video_path", "")
        feature_keys = info.get("features", {})
        for cand in (
            "observation.images.front",
            "observation.images.image",
            "observation.image",
            "image",
        ):
            if cand in feature_keys:
                self.img_key = cand
                break
        else:
            sys.exit(f"No image feature found in info.json features: {list(feature_keys.keys())}")
        self.wrist_key: Optional[str] = None
        if include_wrist:
            for cand in (
                "observation.images.wrist",
                "observation.images.wrist_image",
                "observation.images.eye_in_hand",
                "observation.wrist_image",
                "wrist_image",
                "wrist",
            ):
                if cand in feature_keys:
                    self.wrist_key = cand
                    break
        feature = feature_keys[self.img_key]
        self.is_video = feature.get("dtype") == "video" or self.video_template != ""
        self.per_episode_video = (
            "episode_index" in self.video_template or "{episode_chunk:" in self.video_template
        )
        print(
            f"image source: {('mp4 video' if self.is_video else 'embedded parquet')} (third={self.img_key}, wrist={self.wrist_key or 'NONE'}, per_episode={self.per_episode_video})"
        )
        self._video_cache: dict = {}
        self._video_cache_wrist: dict = {}
        self._cache_max = 4

    def has_wrist(self) -> bool:
        return self.wrist_key is not None

    def get_frame_from_parquet_row(self, row: pd.Series, key: Optional[str] = None) -> np.ndarray:
        k = key or self.img_key
        v = row[k]
        if isinstance(v, dict):
            v = v.get("bytes")
        if isinstance(v, (bytes, bytearray, memoryview)):
            img = Image.open(io.BytesIO(v)).convert("RGB")
            return np.array(img)
        sys.exit(f"Unexpected image type in parquet[{k}]: {type(v)}")

    def _load_video(self, video_path: Path):
        try:
            import imageio.v3 as iio

            frames = iio.imread(str(video_path))
        except Exception as e:
            sys.exit(f"Failed to read {video_path}: {e}")
        return frames

    def _format_video_path(self, key: Optional[str] = None, **kwargs) -> Path:
        video_key = key or self.img_key
        try:
            rel = self.video_template.format(video_key=video_key, **kwargs)
        except KeyError as e:
            sys.exit(
                f"video_template {self.video_template!r} requires {e}; provided {list(kwargs.keys())}"
            )
        return self.root / rel

    def get_frame_per_episode(
        self,
        episode_index: int,
        episode_chunk: int,
        within_episode_idx: int,
        key: Optional[str] = None,
    ) -> np.ndarray:
        video_key = key or self.img_key
        cache = self._video_cache_wrist if video_key == self.wrist_key else self._video_cache
        cache_key = ("ep", episode_chunk, episode_index)
        if cache_key not in cache:
            if len(cache) >= self._cache_max:
                oldest = next(iter(cache))
                del cache[oldest]
            video_path = self._format_video_path(
                key=video_key, episode_chunk=episode_chunk, episode_index=episode_index
            )
            cache[cache_key] = self._load_video(video_path)
        return cache[cache_key][within_episode_idx]

    def get_frame_per_chunk(
        self, chunk_index: int, file_index: int, within_file_idx: int, key: Optional[str] = None
    ) -> np.ndarray:
        video_key = key or self.img_key
        cache = self._video_cache_wrist if video_key == self.wrist_key else self._video_cache
        cache_key = ("ch", chunk_index, file_index)
        if cache_key not in cache:
            if len(cache) >= self._cache_max:
                oldest = next(iter(cache))
                del cache[oldest]
            video_path = self._format_video_path(
                key=video_key, chunk_index=chunk_index, file_index=file_index
            )
            cache[cache_key] = self._load_video(video_path)
        return cache[cache_key][within_file_idx]


def build_proprio(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=np.float32)
    if s.shape[0] == 8:
        return s
    if s.shape[0] == 9:
        return np.concatenate([s[:3], quat2axisangle(s[3:7]), s[7:9]]).astype(np.float32)
    sys.exit(f"Unsupported state dimension: {s.shape[0]}")


def convert_episode(
    ep_df: pd.DataFrame,
    img_provider: ImageProvider,
    task_id: int,
    demo_id: int,
    instruction: str,
    suite_name: str,
    out_dir: Path,
    f_jsonl,
    args: argparse.Namespace,
) -> int:
    ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
    T = len(ep_df)
    if T < args.chunk_size + 1:
        return 0
    in_video = img_provider.is_video
    if in_video:
        if img_provider.per_episode_video:
            ep_index_global = int(ep_df["episode_index"].iloc[0])
            ep_chunk = ep_index_global // 1000
            within_offsets = ep_df["frame_index"].astype(int).values
        else:
            for cand in ("video_chunk_index", "data_chunk_index", "chunk_index"):
                if cand in ep_df.columns:
                    video_chunk = int(ep_df[cand].iloc[0])
                    break
            else:
                video_chunk = 0
            for cand in ("video_file_index", "data_file_index", "file_index"):
                if cand in ep_df.columns:
                    video_file = int(ep_df[cand].iloc[0])
                    break
            else:
                video_file = 0
            within_offsets = ep_df["frame_index"].astype(int).values
    n_samples = 0
    img_buffer: deque[str] = deque(maxlen=args.num_images)
    wrist_buffer: deque[str] = deque(maxlen=args.num_images)
    has_wrist = img_provider.has_wrist() and (not args.no_wrist)
    actions = np.stack(ep_df["action"].values).astype(np.float32)
    for t in range(T):
        row = ep_df.iloc[t]
        if in_video:
            if img_provider.per_episode_video:
                frame = img_provider.get_frame_per_episode(
                    episode_index=ep_index_global,
                    episode_chunk=ep_chunk,
                    within_episode_idx=int(within_offsets[t]),
                )
            else:
                frame = img_provider.get_frame_per_chunk(
                    chunk_index=video_chunk,
                    file_index=video_file,
                    within_file_idx=int(within_offsets[t]),
                )
        else:
            frame = img_provider.get_frame_from_parquet_row(row)
        frame = to_256(frame)
        rel = f"images/task{task_id:02d}_demo{demo_id:03d}/f{t:05d}.png"
        save_image(frame, out_dir / rel)
        img_buffer.append(rel)
        if has_wrist:
            if in_video:
                if img_provider.per_episode_video:
                    wframe = img_provider.get_frame_per_episode(
                        episode_index=ep_index_global,
                        episode_chunk=ep_chunk,
                        within_episode_idx=int(within_offsets[t]),
                        key=img_provider.wrist_key,
                    )
                else:
                    wframe = img_provider.get_frame_per_chunk(
                        chunk_index=video_chunk,
                        file_index=video_file,
                        within_file_idx=int(within_offsets[t]),
                        key=img_provider.wrist_key,
                    )
            else:
                wframe = img_provider.get_frame_from_parquet_row(row, key=img_provider.wrist_key)
            wframe = to_256(wframe)
            wrel = f"images/task{task_id:02d}_demo{demo_id:03d}/f{t:05d}_wrist.png"
            save_image(wframe, out_dir / wrel)
            wrist_buffer.append(wrel)
        proprio = build_proprio(row["observation.state"])
        end = min(t + args.chunk_size, T)
        chunk = actions[t:end]
        if len(chunk) < args.chunk_size:
            pad = np.tile(chunk[-1:], (args.chunk_size - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        if args.gripper_to_rlds:
            chunk = chunk.copy()
            chunk[:, 6] = (chunk[:, 6] + 1.0) / 2.0
        padded_imgs = pad_history(list(img_buffer), args.num_images)
        padded_wrist = pad_history(list(wrist_buffer), args.num_images) if has_wrist else []
        sample = {
            "images": padded_imgs,
            "proprio": proprio.tolist(),
            "instruction": instruction,
            "action_chunk": chunk.tolist(),
            "chunk_size": args.chunk_size,
            "suite": suite_name,
            "task_id": task_id,
            "demo_id": demo_id,
            "step": t,
        }
        if has_wrist:
            sample["wrist_images"] = padded_wrist
        f_jsonl.write(json.dumps(sample, ensure_ascii=False) + "\n")
        n_samples += 1
    return n_samples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lerobot_root",
        type=str,
        required=True,
        help="Path to unzipped LeRobot dataset directory.",
    )
    p.add_argument(
        "--suite_name",
        type=str,
        required=True,
        help="Suite name to embed in output JSONL (e.g. libero_plus_goal).",
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--num_demos_per_task", type=int, default=None, help="Cap demos per task. Default: all."
    )
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument(
        "--gripper_to_rlds",
        action="store_true",
        default=False,
        help="Convert action gripper from {-1,+1} → {0,1} ",
    )
    p.add_argument(
        "--max_episodes", type=int, default=None, help="Maximum number of episodes to process."
    )
    p.add_argument(
        "--no_wrist",
        action="store_true",
        default=False,
        help="Disable wrist-camera extraction and write single-view samples.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = find_dataset_root(Path(args.lerobot_root).expanduser().resolve())
    info = json.load(open(root / "meta" / "info.json"))
    print(f"codebase_version: {info.get('codebase_version')}")
    print(f"total_episodes:   {info.get('total_episodes')}")
    print(f"total_frames:     {info.get('total_frames')}")
    print(f"total_tasks:      {info.get('total_tasks')}")
    tasks_map = load_tasks(root)
    print(f"#tasks loaded:    {len(tasks_map)}")
    img_provider = ImageProvider(root, info, include_wrist=not args.no_wrist)
    out_dir = Path(args.out_dir).expanduser().resolve() / args.suite_name
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "data.jsonl"
    f_jsonl = open(jsonl_path, "w", encoding="utf-8")
    print(f"\nWriting → {jsonl_path}")
    sorted_task_indices = sorted(tasks_map.keys())
    task_remap = {t: i for i, t in enumerate(sorted_task_indices)}
    demos_per_task = defaultdict(int)
    total_samples = 0
    total_episodes = 0
    t_start = time.time()
    for path, df in load_data_iter(root):
        for ep_idx, ep_df in df.groupby("episode_index"):
            if args.max_episodes is not None and total_episodes >= args.max_episodes:
                break
            task_idx_raw = int(ep_df["task_index"].iloc[0])
            task_id = task_remap.get(task_idx_raw)
            if task_id is None:
                continue
            instruction = tasks_map[task_idx_raw]
            if (
                args.num_demos_per_task is not None
                and demos_per_task[task_id] >= args.num_demos_per_task
            ):
                continue
            demo_id = demos_per_task[task_id]
            demos_per_task[task_id] += 1
            n = convert_episode(
                ep_df,
                img_provider,
                task_id,
                demo_id,
                instruction,
                args.suite_name,
                out_dir,
                f_jsonl,
                args,
            )
            total_samples += n
            total_episodes += 1
            if total_episodes % 50 == 0:
                elapsed = time.time() - t_start
                print(
                    f"[{time.strftime('%H:%M:%S')}] episodes={total_episodes:>5}  samples={total_samples:>7}  elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )
    f_jsonl.close()
    print(
        f"\nDone in {(time.time() - t_start) / 60:.1f}min: {total_episodes} episodes, {total_samples} samples → {jsonl_path}"
    )
    print(
        f"Demos per task — min={min(demos_per_task.values(), default=0)}, max={max(demos_per_task.values(), default=0)}, #tasks={len(demos_per_task)}"
    )


if __name__ == "__main__":
    main()
