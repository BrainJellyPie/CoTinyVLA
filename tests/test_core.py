from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_training_module():
    if "transformers" not in sys.modules:
        module = types.ModuleType("transformers")

        class Trainer:
            pass

        module.AutoModelForImageTextToText = object
        module.AutoProcessor = object
        module.Trainer = Trainer
        module.set_seed = lambda seed: None
        sys.modules["transformers"] = module
    return load_module("train_policy", SCRIPTS / "train_policy.py")


def test_training_components(tmp_path: Path):
    train = load_training_module()
    assert train.build_assistant_cot_text("p", "t") == (
        "<|cot_plan|>p<|cot_plan_end|><|cot_think|>t<|cot_think_end|>"
    )

    image = Image.new("RGB", (16, 16))
    messages = train.build_messages_dual([image, image], [image, image], "move")
    assert len(messages) == 2
    assert train.pad_history([1], 3) == [1, 1, 1]
    assert train.ActionHead(8, 16, 7, 4)(torch.zeros(2, 8)).shape == (2, 4, 7)
    assert train.ProprioProjector(8, 16)(torch.zeros(2, 8)).shape == (2, 16)

    for name in ["a.png", "b.png", "wa.png", "wb.png"]:
        image.save(tmp_path / name)
    row = {
        "images": ["a.png", "b.png"],
        "wrist_images": ["wa.png", "wb.png"],
        "proprio": [0.0] * 8,
        "instruction": "move",
        "action_chunk": [[0.0] * 7] * 2,
        "plan_text": "1. move",
        "think_text": "Phase: step 1\nGripper: OPEN\nNext: move",
    }
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = train.ActionDistillDataset(
        str(jsonl), num_images=2, chunk_size=2, proprio_dim=8, num_wrist_images=2
    )
    assert dataset[0]["action_labels"].shape == (2, 7)


def test_conversion_and_label_helpers():
    converter = load_module("convert_libero_plus", SCRIPTS / "convert_libero_plus.py")
    labels = load_module(
        "generate_reasoning_labels", SCRIPTS / "generate_reasoning_labels.py"
    )

    assert converter.pad_history([1], 3) == [1, 1, 1]
    assert converter.build_proprio(np.zeros(8, dtype=np.float32)).shape == (8,)
    assert np.allclose(
        converter.quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3)
    )
    assert labels.clean_plan_output("Plan:\n1. approach\n2. grasp") == (
        "1. approach\n2. grasp"
    )
    assert labels.clean_think_output(
        "Phase: step 1\nGripper: OPEN\nNext: move"
    ) == "Phase: step 1\nGripper: OPEN\nNext: move"


def test_checkpoint_selection_and_aggregation(tmp_path: Path):
    evaluator = load_module("evaluate_libero_plus", SCRIPTS / "evaluate_libero_plus.py")
    for step in [10, 20]:
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "action_head_config.json").write_text(
            json.dumps(
                {
                    "num_images": 8,
                    "num_wrist_images": 8,
                    "uses_cot": True,
                    "model_id": "model",
                }
            ),
            encoding="utf-8",
        )
        (checkpoint / "student_state_dict.pt").write_bytes(b"x")
        (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    checkpoint, processor, report = evaluator.discover_checkpoint(None, [tmp_path])
    assert checkpoint.name == "checkpoint-20"
    assert processor == checkpoint
    assert report["selected"]["global_step"] == 20

    database = tmp_path / "jobs.sqlite3"
    stats = evaluator.initialize_queue(
        database,
        ["libero_spatial", "libero_goal"],
        {"libero_spatial": 2, "libero_goal": 2},
        None,
        True,
    )
    assert stats["total"] == 4

    connection = evaluator.connect_db(database)
    rows = list(connection.execute("SELECT job_id, suite, task_id FROM jobs ORDER BY job_id"))
    for index, row in enumerate(rows):
        result = {
            "suite": row["suite"],
            "task_id": row["task_id"],
            "success": index % 2 == 0,
            "elapsed_s": 1.0,
            "category": "Layout",
            "difficulty_level": 1,
            "classification_name": "task",
        }
        connection.execute(
            "UPDATE jobs SET status='done', success=?, result_json=? WHERE job_id=?",
            (int(result["success"]), json.dumps(result), row["job_id"]),
        )
    connection.close()

    summary = evaluator.aggregate_outputs(database, tmp_path, ["libero_spatial", "libero_goal"])
    assert summary["valid_tasks"] == 4
    assert (tmp_path / "success_tables.md").exists()
