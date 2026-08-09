import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "examples" / "smolvla_libero_plus_multisuite_lora_local.py"
TASK_CSV = ROOT / "compe" / "t1" / "T1_TASKS.csv"


def _literal_assignments() -> dict[str, object]:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_exact_public_task_ids_are_the_training_objective():
    values = _literal_assignments()
    configured = values["TRACK1_PUBLIC_EVAL_TASKS"]
    csv_rows = list(csv.DictReader(TASK_CSV.open(newline="", encoding="utf-8")))
    expected_pairs = {(row["suite"], int(row["libero_plus_id"])) for row in csv_rows}

    assert set(configured.values()) == expected_pairs
    assert len(configured) == len(csv_rows) == 4
    assert set(configured) == values["TRACK1_TARGET_TASKS"]


def test_target_data_and_eval_match_track1_conditions():
    values = _literal_assignments()

    assert values["TARGET_EPISODES_PER_TASK"] > values["TRAIN_EPISODES_PER_TASK"]
    assert values["EVAL_OBSERVATION_SIZE"] == 128


def test_training_enables_visual_augmentation():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"--dataset.image_transforms.enable=true"' in source
    assert '"--dataset.image_transforms.max_num_transforms=4"' in source
