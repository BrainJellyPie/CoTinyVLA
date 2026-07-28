#!/usr/bin/env python3
"""Train a Qwen3.5 vision-language-action policy with temporal third-person and wrist observations."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, set_seed

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger("train")
SYSTEM_PROMPT = "You are a robot manipulation policy. Given a sequence of observation images (oldest to newest) and a task instruction, predict the next action chunk."
COT_PLAN_OPEN = "<|cot_plan|>"
COT_PLAN_CLOSE = "<|cot_plan_end|>"
COT_THINK_OPEN = "<|cot_think|>"
COT_THINK_CLOSE = "<|cot_think_end|>"
COT_SPECIAL_TOKENS = [COT_PLAN_OPEN, COT_PLAN_CLOSE, COT_THINK_OPEN, COT_THINK_CLOSE]


def build_assistant_cot_text(plan_text: str, think_text: str) -> str:
    return (
        f"{COT_PLAN_OPEN}{plan_text}{COT_PLAN_CLOSE}{COT_THINK_OPEN}{think_text}{COT_THINK_CLOSE}"
    )


def build_messages(pil_images: List[Image.Image], instruction: str) -> List[Dict]:
    image_blocks: List[Dict] = []
    for i, im in enumerate(pil_images):
        label = f"Image {i + 1}"
        if i == 0:
            label += " (oldest)"
        elif i == len(pil_images) - 1:
            label += " (current)"
        image_blocks.append({"type": "text", "text": f"[{label}]"})
        image_blocks.append({"type": "image", "image": im})
    image_blocks.append(
        {"type": "text", "text": f"\nInstruction: {instruction}\nPredict the next action chunk."}
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": image_blocks},
    ]


def build_messages_dual(
    third_images: List[Image.Image], wrist_images: List[Image.Image], instruction: str
) -> List[Dict]:
    if not wrist_images:
        return build_messages(third_images, instruction)
    image_blocks: List[Dict] = []
    N3 = len(third_images)
    for i, im in enumerate(third_images):
        suffix = ""
        if i == 0:
            suffix = " (oldest)"
        elif i == N3 - 1:
            suffix = " (current)"
        image_blocks.append({"type": "text", "text": f"[Third frame {i + 1}{suffix}]"})
        image_blocks.append({"type": "image", "image": im})
    NW = len(wrist_images)
    for i, im in enumerate(wrist_images):
        suffix = ""
        if i == 0:
            suffix = " (oldest)"
        elif i == NW - 1:
            suffix = " (current)"
        image_blocks.append({"type": "text", "text": f"[Wrist frame {i + 1}{suffix}]"})
        image_blocks.append({"type": "image", "image": im})
    image_blocks.append(
        {"type": "text", "text": f"\nInstruction: {instruction}\nPredict the next action chunk."}
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": image_blocks},
    ]


def pad_history(frames: list, n: int) -> list:
    if len(frames) >= n:
        return frames[-n:]
    if len(frames) == 0:
        raise ValueError("pad_history: frames is empty")
    return [frames[0]] * (n - len(frames)) + list(frames)


ACTION_DIM = 7
PROPRIO_DIM = 8


class ProprioProjector(nn.Module):

    def __init__(self, proprio_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.mlp(proprio)


class ActionHead(nn.Module):

    def __init__(self, input_dim: int, hidden_size: int, action_dim: int, chunk_size: int):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        out_dim = action_dim * chunk_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).reshape(-1, self.chunk_size, self.action_dim)


class Qwen35ActionPolicy(nn.Module):

    def __init__(
        self,
        model_id: str,
        action_dim: int = ACTION_DIM,
        proprio_dim: int = PROPRIO_DIM,
        chunk_size: int = 8,
        use_proprio: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "sdpa",
        cot_token_ids: dict | None = None,
    ):
        super().__init__()
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation=attn_impl, trust_remote_code=True
        )
        config = self.vlm.config
        if hasattr(config, "text_config"):
            hidden_size = config.text_config.hidden_size
        else:
            hidden_size = config.hidden_size
        self.hidden_size = hidden_size
        self.use_proprio = use_proprio
        self.proprio_dim = proprio_dim
        if cot_token_ids is not None:
            for name, tid in cot_token_ids.items():
                self.register_buffer(
                    f"_cot_id_{name}", torch.tensor(int(tid), dtype=torch.long), persistent=True
                )
        self._cot_token_ids = cot_token_ids
        if hasattr(self.vlm, "lm_head"):
            for p in self.vlm.lm_head.parameters():
                p.requires_grad = False
        if use_proprio:
            self.proprio_projector = ProprioProjector(proprio_dim, hidden_size)
            head_input_dim = 2 * hidden_size
        else:
            self.proprio_projector = None
            head_input_dim = hidden_size
        self.action_head = ActionHead(head_input_dim, hidden_size, action_dim, chunk_size)
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        action_labels: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        action_target_pos: torch.Tensor | None = None,
        action_loss_weight: float = 10.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        vlm_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        for name, val in [
            ("pixel_values", pixel_values),
            ("image_grid_thw", image_grid_thw),
            ("mm_token_type_ids", mm_token_type_ids),
            ("video_grid_thw", video_grid_thw),
            ("second_per_grid_ts", second_per_grid_ts),
        ]:
            if val is not None:
                vlm_kwargs[name] = val
        outputs = self.vlm(**vlm_kwargs)
        last_hidden = outputs.hidden_states[-1]
        if action_target_pos is not None:
            batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            target_hidden = last_hidden[batch_idx, action_target_pos]
        else:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            target_hidden = last_hidden[batch_idx, seq_lengths]
        if self.use_proprio and proprio is not None:
            proprio = proprio.to(target_hidden.dtype)
            proprio_emb = self.proprio_projector(proprio)
            features = torch.cat([target_hidden, proprio_emb], dim=-1)
        else:
            features = target_hidden
        pred_actions = self.action_head(features)
        result = {"pred_actions": pred_actions}
        action_loss = None
        if action_labels is not None:
            action_loss = nn.functional.l1_loss(pred_actions, action_labels)
            result["action_loss"] = action_loss
        lm_loss = None
        if labels is not None and hasattr(self.vlm, "lm_head"):
            shift_hidden = last_hidden[:, :-1, :]
            shift_labels = labels[:, 1:]
            sup_mask = shift_labels != -100
            n_sup = int(sup_mask.sum().item())
            if n_sup > 0:
                sup_hidden = shift_hidden[sup_mask]
                sup_targets = shift_labels[sup_mask]
                sup_logits = self.vlm.lm_head(sup_hidden)
                lm_loss = nn.functional.cross_entropy(sup_logits, sup_targets, reduction="mean")
                result["lm_loss"] = lm_loss
        total = None
        if action_loss is not None:
            total = action_loss_weight * action_loss
        if lm_loss is not None:
            lm_w = kwargs.get("lm_loss_weight", 0.1)
            total = total + lm_w * lm_loss if total is not None else lm_w * lm_loss
        if total is not None:
            result["loss"] = total
        return result

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        if hasattr(self.vlm, "config"):
            self.vlm.config.use_cache = False

    def gradient_checkpointing_disable(self):
        self.vlm.gradient_checkpointing_disable()


class ActionDistillDataset(Dataset):

    def __init__(
        self,
        jsonl_path: str,
        num_images: int = 8,
        chunk_size: int = 8,
        proprio_dim: int = PROPRIO_DIM,
        num_wrist_images: int = 0,
        instruction_variants_json: str | None = None,
        variant_prob: float = 0.0,
        seed: int = 42,
    ):
        self.jsonl_path = str(Path(jsonl_path).resolve())
        self.base_dir = Path(jsonl_path).resolve().parent
        self.num_images = num_images
        self.num_wrist_images = num_wrist_images
        self.chunk_size = chunk_size
        self.proprio_dim = proprio_dim
        self.offsets: List[int] = []
        n_with_wrist = 0
        needle_present = b'"wrist_images"'
        needle_empty = b'"wrist_images": []'
        with open(self.jsonl_path, "rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                self.offsets.append(off)
                if needle_present in line and needle_empty not in line:
                    n_with_wrist += 1
        n = len(self.offsets)
        log.info(f"Indexed {n} samples from {self.jsonl_path} (streaming mode)")
        if self.num_wrist_images > 0:
            log.info(
                f"  rows with wrist_images: {n_with_wrist}/{n} (num_wrist_images={self.num_wrist_images}); missing rows fall back to single-view"
            )
        self.variant_map: dict | None = None
        self.variant_prob = float(variant_prob)
        if instruction_variants_json and Path(instruction_variants_json).exists():
            with open(instruction_variants_json) as f:
                self.variant_map = json.load(f)
            n_keys = len(self.variant_map)
            avg_var = sum((len(v) for v in self.variant_map.values())) // max(n_keys, 1)
            log.info(
                f"[dataset] loaded {n_keys} base instructions × {avg_var} variants (variant_prob={self.variant_prob})"
            )
        self._rng = random.Random(seed)
        self._fh = None
        self._fh_pid = -1

    def __len__(self) -> int:
        return len(self.offsets)

    def _get_fh(self):
        pid = os.getpid()
        if self._fh is None or self._fh_pid != pid:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = open(self.jsonl_path, "rb")
            self._fh_pid = pid
        return self._fh

    def _resolve_image_paths(self, img_rels: list, suite: str | None) -> list:
        out = []
        for img_rel in img_rels:
            if suite is not None:
                primary = self.base_dir / suite / img_rel
                if primary.exists():
                    out.append(primary)
                else:
                    out.append(self.base_dir / img_rel)
            else:
                out.append(self.base_dir / img_rel)
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fh = self._get_fh()
        fh.seek(self.offsets[idx])
        line = fh.readline()
        row = json.loads(line)
        suite = row.get("suite")
        third_paths = self._resolve_image_paths(row["images"][: self.num_images], suite)
        pil_third = []
        for img_path in third_paths:
            with Image.open(img_path) as im:
                pil_third.append(im.convert("RGB"))
        pil_wrist: List[Image.Image] = []
        if self.num_wrist_images > 0 and row.get("wrist_images"):
            wrist_paths = self._resolve_image_paths(
                row["wrist_images"][: self.num_wrist_images], suite
            )
            for img_path in wrist_paths:
                with Image.open(img_path) as im:
                    pil_wrist.append(im.convert("RGB"))
        chunk = row["action_chunk"][: self.chunk_size]
        while len(chunk) < self.chunk_size:
            chunk.append(chunk[-1])
        action_labels = np.array(chunk, dtype=np.float32)
        proprio = np.array(row.get("proprio", [0.0] * self.proprio_dim), dtype=np.float32)[
            : self.proprio_dim
        ]
        if proprio.shape[0] < self.proprio_dim:
            proprio = np.pad(proprio, (0, self.proprio_dim - proprio.shape[0]))
        instruction = row["instruction"]
        if self.variant_map is not None and self.variant_prob > 0.0:
            if self._rng.random() < self.variant_prob:
                variants = self.variant_map.get(instruction)
                if variants:
                    instruction = self._rng.choice(variants)
        messages = build_messages_dual(pil_third, pil_wrist, instruction)
        return {
            "messages": messages,
            "action_labels": action_labels,
            "proprio": proprio,
            "num_images": len(pil_third) + len(pil_wrist),
            "plan_text": row.get("plan_text", ""),
            "think_text": row.get("think_text", ""),
        }


@dataclass
class ActionCollator:
    processor: Any
    max_length: int
    cot_token_ids: dict

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        prompt_texts = []
        full_texts = []
        all_images: List[List[Image.Image]] = []
        action_labels = []
        proprios = []
        for ex in examples:
            prompt_text = self.processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=True
            ).strip()
            assistant_text = build_assistant_cot_text(
                ex.get("plan_text", "") or "(no plan)", ex.get("think_text", "") or "(no think)"
            )
            full_text = prompt_text + assistant_text
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            images = []
            for msg in ex["messages"]:
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        images.append(item["image"].convert("RGB"))
            all_images.append(images)
            action_labels.append(torch.from_numpy(ex["action_labels"]))
            proprios.append(torch.from_numpy(ex["proprio"]))
        batch = self.processor(
            text=full_texts,
            images=all_images if any(all_images) else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        B, T = input_ids.shape
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        plan_open_id = self.cot_token_ids["plan_open"]
        think_close_id = self.cot_token_ids["think_close"]
        action_target_pos = torch.zeros(B, dtype=torch.long)
        for i in range(B):
            row = input_ids[i]
            plan_pos = (row == plan_open_id).nonzero(as_tuple=False).flatten()
            if plan_pos.numel() > 0:
                plan_start = int(plan_pos[0].item())
                labels[i, :plan_start] = -100
            else:
                labels[i, :] = -100
            tc_pos = (row == think_close_id).nonzero(as_tuple=False).flatten()
            if tc_pos.numel() > 0:
                action_target_pos[i] = int(tc_pos[-1].item())
            else:
                action_target_pos[i] = int(attention_mask[i].sum().item() - 1)
        batch["labels"] = labels
        batch["action_target_pos"] = action_target_pos
        batch["action_labels"] = torch.stack(action_labels)
        batch["proprio"] = torch.stack(proprios)
        return batch


class ActionTrainer(Trainer):

    def __init__(
        self,
        *args,
        proprio_lr_mult: float = 5.0,
        action_head_lr_mult: float = 5.0,
        action_loss_weight: float = 1.0,
        lm_loss_weight: float = 0.1,
        **kwargs,
    ):
        self._proprio_lr_mult = float(proprio_lr_mult)
        self._action_head_lr_mult = float(action_head_lr_mult)
        self._action_loss_weight = float(action_loss_weight)
        self._lm_loss_weight = float(lm_loss_weight)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch: int | None = None, **kwargs
    ):
        action_labels = inputs.pop("action_labels")
        labels = inputs.pop("labels", None)
        action_target_pos = inputs.pop("action_target_pos", None)
        outputs = model(
            **inputs,
            action_labels=action_labels,
            labels=labels,
            action_target_pos=action_target_pos,
            action_loss_weight=self._action_loss_weight,
            lm_loss_weight=self._lm_loss_weight,
        )
        loss = outputs["loss"]
        if "lm_loss" in outputs and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            try:
                lm_v = float(outputs["lm_loss"].detach())
                act_v = float(outputs.get("action_loss", torch.tensor(0.0)).detach())
                log.info(
                    f"  step={self.state.global_step}  total={float(loss):.4f}  lm={lm_v:.4f}  action_l1={act_v:.4f}"
                )
            except Exception:
                pass
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
        base_lr = self.args.learning_rate
        wd = self.args.weight_decay
        decay_names = set(self.get_decay_parameter_names(opt_model))

        def classify(pname: str) -> str:
            if pname.startswith("proprio_projector") or ".proprio_projector." in pname:
                return "proprio"
            if pname.startswith("action_head") or ".action_head." in pname:
                return "head"
            return "vlm"

        lr_of = {
            "vlm": base_lr,
            "proprio": base_lr * self._proprio_lr_mult,
            "head": base_lr * self._action_head_lr_mult,
        }
        buckets: Dict[tuple, List[torch.nn.Parameter]] = {
            (g, d): [] for g in lr_of for d in (True, False)
        }
        stats = {g: 0 for g in lr_of}
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            g = classify(name)
            d = name in decay_names
            buckets[g, d].append(param)
            stats[g] += param.numel()
        grouped = []
        for (g, d), params in buckets.items():
            if not params:
                continue
            grouped.append({"params": params, "lr": lr_of[g], "weight_decay": wd if d else 0.0})
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop("lr", None)
        optimizer_kwargs.pop("weight_decay", None)
        self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
        total = sum(stats.values())
        tlog = logging.getLogger("train")
        tlog.info("Differential LR groups:")
        for g, nparams in stats.items():
            pct = 100 * nparams / max(total, 1)
            tlog.info(f"  {g:8s}  params={nparams:>12,} ({pct:4.1f}%)  lr={lr_of[g]:.2e}")
        return self.optimizer

    def _save(self, output_dir=None, state_dict=None):
        import json
        import os

        out = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(out, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(out, "pytorch_model.bin"))
        torch.save(state_dict, os.path.join(out, "student_state_dict.pt"))
        torch.save(self.args, os.path.join(out, "training_args.bin"))
        meta = getattr(self, "_action_meta", None)
        if meta is not None:
            with open(os.path.join(out, "action_head_config.json"), "w") as f:
                json.dump(meta, f, indent=2)

    def log(self, logs, *args, **kwargs):
        super().log(logs, *args, **kwargs)
        if "loss" in logs:
            step = self.state.global_step
            lr = logs.get("learning_rate", 0)
            logging.getLogger("train").info(f"step={step}  loss={logs['loss']:.4f}  lr={lr:.2e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-0.8B")
    p.add_argument(
        "--resume_from_state_dict",
        type=str,
        default=None,
        help="Load model weights from student_state_dict.pt.",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume model, optimizer, scheduler, and training state from a Trainer checkpoint.",
    )
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--eval_jsonl", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument(
        "--num_images", type=int, default=8, help="Per-view third-person history depth (N)."
    )
    p.add_argument(
        "--num_wrist_images",
        type=int,
        default=0,
        help="Number of wrist-camera history frames. Set to 0 for a single camera.",
    )
    p.add_argument(
        "--instruction_variants_json",
        type=str,
        default=None,
        help="JSON mapping each instruction to a list of paraphrases.",
    )
    p.add_argument(
        "--variant_prob",
        type=float,
        default=0.0,
        help="Probability of sampling a paraphrased instruction.",
    )
    p.add_argument("--action_dim", type=int, default=ACTION_DIM)
    p.add_argument("--proprio_dim", type=int, default=PROPRIO_DIM)
    p.add_argument(
        "--no_proprio", action="store_true", help="Disable proprio projector (use only image+text)"
    )
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=1e-05)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument(
        "--proprio_lr_mult",
        type=float,
        default=5.0,
        help="LR multiplier for ProprioProjector (random init)",
    )
    p.add_argument(
        "--action_head_lr_mult",
        type=float,
        default=5.0,
        help="LR multiplier for ActionHead (random init)",
    )
    p.add_argument(
        "--action_loss_weight",
        type=float,
        default=1.0,
        help="Weight on action L1. Total = w_a × L1 + w_lm × CE.",
    )
    p.add_argument(
        "--lm_loss_weight",
        type=float,
        default=0.1,
        help="Weight on language-model cross-entropy. Set 0 to disable LM supervision (action-only baseline).",
    )
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--optim", type=str, default="adamw_torch", choices=["adamw_torch", "adamw_bnb_8bit"]
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes per training process.",
    )
    p.add_argument("--freeze_vision", action="store_true", help="Freeze vision-encoder parameters.")
    p.add_argument("--measure_vram_only", action="store_true")
    return p.parse_args()


def pick_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def vram_probe(model, collator, dataset, dtype, batch_size=1):
    if not torch.cuda.is_available():
        log.warning("VRAM probe skipped: no CUDA")
        return
    device = torch.device("cuda")
    model.train()
    model.to(device)
    n = min(batch_size, len(dataset))
    samples = [dataset[i] for i in range(n)]
    batch = collator(samples)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    log.info(f"VRAM probe — using batch_size={n}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=dtype):
        out = model(**batch)
        loss = out["loss"]
    loss.backward()
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    log.info(f"VRAM probe — loss: {loss.item():.6f}, peak: {peak:.3f} GiB")
    model.zero_grad(set_to_none=True)


def main():
    args = parse_args()
    if args.resume_from_state_dict and args.resume_from_checkpoint:
        raise ValueError("Choose only one resume option")
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = pick_dtype()
    log.info(
        f"Config — dtype={dtype}, chunk_size={args.chunk_size}, num_images={args.num_images}, num_wrist_images={args.num_wrist_images}, action_dim={args.action_dim}"
    )
    log.info(f"Loading processor: {args.model_id}")
    proc_kwargs = {"trust_remote_code": True}
    if args.min_pixels is not None:
        proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
    tok = processor.tokenizer
    n_added = tok.add_special_tokens({"additional_special_tokens": COT_SPECIAL_TOKENS})
    cot_token_ids = {
        "plan_open": tok.convert_tokens_to_ids(COT_PLAN_OPEN),
        "plan_close": tok.convert_tokens_to_ids(COT_PLAN_CLOSE),
        "think_open": tok.convert_tokens_to_ids(COT_THINK_OPEN),
        "think_close": tok.convert_tokens_to_ids(COT_THINK_CLOSE),
    }
    log.info(f"Added {n_added} new CoT tokens. ids = {cot_token_ids}")
    use_proprio = not args.no_proprio
    log.info(f"Loading model: {args.model_id}  (use_proprio={use_proprio})")
    t0 = time.time()
    model = Qwen35ActionPolicy(
        model_id=args.model_id,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
        chunk_size=args.chunk_size,
        use_proprio=use_proprio,
        dtype=dtype,
        attn_impl=args.attn_implementation,
        cot_token_ids=cot_token_ids,
    )
    if n_added > 0:
        new_size = len(tok)
        log.info(
            f"Resizing embeddings: {model.vlm.get_input_embeddings().num_embeddings} → {new_size}"
        )
        model.vlm.resize_token_embeddings(new_size)
    if hasattr(model.vlm, "lm_head"):
        for p in model.vlm.lm_head.parameters():
            p.requires_grad = True
        log.info("Unfroze lm_head for CoT language-model supervision.")
    log.info(f"Model loaded in {time.time() - t0:.1f}s (hidden={model.hidden_size})")
    if args.resume_from_state_dict:
        sd_path = Path(args.resume_from_state_dict)
        if sd_path.is_dir():
            sd_path = sd_path / "student_state_dict.pt"
        log.info(f"Resuming student weights from: {sd_path}")
        try:
            sd = torch.load(sd_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(sd_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            log.warning(f"Missing keys: {len(missing)} (first 3: {missing[:3]})")
        if unexpected:
            log.warning(f"Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
        log.info("Student weights loaded successfully for continued training.")
    if args.freeze_vision:
        cnt = 0
        for name, p in model.vlm.named_parameters():
            if "visual" in name or "vision" in name:
                p.requires_grad = False
                cnt += 1
        log.info(f"Froze {cnt} vision params")
    if args.gradient_checkpointing:
        model.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model.vlm, "config"):
            model.vlm.config.use_cache = False
    total = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    head_params = sum((p.numel() for p in model.action_head.parameters()))
    proj_params = (
        sum((p.numel() for p in model.proprio_projector.parameters()))
        if model.proprio_projector is not None
        else 0
    )
    log.info(
        f"Params — total: {total:,}  trainable: {trainable:,}  action_head: {head_params:,}  proprio_proj: {proj_params:,}"
    )
    log.info(f"Loading dataset: {args.train_jsonl}")
    train_ds = ActionDistillDataset(
        args.train_jsonl,
        args.num_images,
        args.chunk_size,
        args.proprio_dim,
        num_wrist_images=args.num_wrist_images,
        instruction_variants_json=args.instruction_variants_json,
        variant_prob=args.variant_prob,
        seed=args.seed,
    )
    eval_ds = (
        ActionDistillDataset(
            args.eval_jsonl,
            args.num_images,
            args.chunk_size,
            args.proprio_dim,
            num_wrist_images=args.num_wrist_images,
            instruction_variants_json=None,
            variant_prob=0.0,
            seed=args.seed,
        )
        if args.eval_jsonl
        else None
    )
    collator = ActionCollator(
        processor=processor, max_length=args.max_length, cot_token_ids=cot_token_ids
    )
    log.info(
        f"Train: {len(train_ds)} samples" + (f", Eval: {len(eval_ds)} samples" if eval_ds else "")
    )
    if args.measure_vram_only:
        vram_probe(model, collator, train_ds, dtype, batch_size=args.per_device_train_batch_size)
        return
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        remove_unused_columns=False,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        optim=args.optim,
    )
    trainer = ActionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        proprio_lr_mult=args.proprio_lr_mult,
        action_head_lr_mult=args.action_head_lr_mult,
        action_loss_weight=args.action_loss_weight,
        lm_loss_weight=args.lm_loss_weight,
    )
    trainer._action_meta = {
        "action_dim": args.action_dim,
        "chunk_size": args.chunk_size,
        "num_images": args.num_images,
        "num_wrist_images": args.num_wrist_images,
        "proprio_dim": args.proprio_dim,
        "use_proprio": use_proprio,
        "hidden_size": model.hidden_size,
        "model_id": args.model_id,
        "cot_token_ids": cot_token_ids,
        "vocab_size": len(tok),
        "uses_cot": True,
    }
    log.info("Training start")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    log.info(f"Training done in {time.time() - t0:.0f}s")
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        torch.save(unwrapped.state_dict(), Path(args.output_dir) / "student_state_dict.pt")
        with open(Path(args.output_dir) / "action_head_config.json", "w") as handle:
            json.dump(trainer._action_meta, handle, indent=2)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
