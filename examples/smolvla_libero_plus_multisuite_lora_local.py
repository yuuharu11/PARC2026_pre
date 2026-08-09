"""
Local SmolVLA LoRA training across all 40 LIBERO-plus training tasks.

This script extends the local port of examples/smolvla_libero_spatial_lora.ipynb
from the 10 Spatial tasks to all 40 available Spatial/Object/Goal/libero_10
task instructions. It selects an equal number of evenly spaced episodes per
task, then evaluates before/after performance across all four suites by
matching the 40 trained task names to a representative perturbation-variant
task ID in their actual suite (each suite's task list is hundreds of
perturbation variants per base instruction, not distinct instructions -- see
the comment above CATEGORY_PREFERENCE). Colab-only operations and bare-venv
compatibility fixes from smolvla_libero_spatial_lora_local.py are preserved;
see examples/README.md.

Requires: a Python 3.12 venv with torch==2.9.1+cu126,
torchvision==0.24.1+cu126, torchcodec==0.9.1, and mujoco==3.7.0
preinstalled (see examples/README.md), run as root with network access
(apt-get, git clone, HF downloads) and ~20GB free disk under /content.
"""

import datetime as _dt
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["HF_LEROBOT_HOME"] = "/content/lerobot_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def phase(name: str) -> None:
    print(f"\n===== [{_dt.datetime.now().isoformat(timespec='seconds')}] {name} =====", flush=True)


if sys.version_info < (3, 12):
    raise RuntimeError("Python 3.12以上が必要です。")

if not torch.cuda.is_available():
    raise RuntimeError("GPUランタイムを選択してください。")

phase("1. Runtime check")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# 2. System packages
# ---------------------------------------------------------------------------
phase("2. System packages")


def run_quiet(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if check and result.returncode != 0:
        raise RuntimeError(result.stdout[-6000:])

    return result


run_quiet(["apt-get", "update", "-qq"])
run_quiet(
    [
        "apt-get",
        "install",
        "-y",
        "-qq",
        "ffmpeg",
        "git",
        "unzip",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libexpat1",
        "libfontconfig1-dev",
        "libmagickwand-dev",
    ]
)

print("System packages ready.")

# ---------------------------------------------------------------------------
# 3. Install LeRobot
# ---------------------------------------------------------------------------
phase("3. Install LeRobot v0.6.0")

LEROBOT_TAG = "v0.6.0"
LEROBOT_DIR = Path("/content/lerobot")
LEROBOT_SRC = LEROBOT_DIR / "src"

run_quiet([sys.executable, "-m", "pip", "uninstall", "-y", "lerobot", "torchao"], check=False)

shutil.rmtree(LEROBOT_DIR, ignore_errors=True)

run_quiet(
    [
        "git",
        "clone",
        "--quiet",
        "--depth",
        "1",
        "--branch",
        LEROBOT_TAG,
        "https://github.com/huggingface/lerobot.git",
        str(LEROBOT_DIR),
    ]
)

smolvlm_source = LEROBOT_SRC / "lerobot" / "policies" / "smolvla" / "smolvlm_with_expert.py"

if not torch.cuda.is_bf16_supported():
    source = smolvlm_source.read_text(encoding="utf-8")
    source = source.replace('torch_dtype="bfloat16",', 'torch_dtype="float16",', 1)
    smolvlm_source.write_text(source, encoding="utf-8")

train_script = LEROBOT_SRC / "lerobot" / "scripts" / "lerobot_train.py"
source = train_script.read_text(encoding="utf-8")
source = source.replace("logging.info(pformat(cfg.to_dict()))", "logging.debug(pformat(cfg.to_dict()))", 1)
source = source.replace("disable=inside_slurm(),", "disable=True,", 1)
train_script.write_text(source, encoding="utf-8")

run_quiet(
    [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "-e", f"{LEROBOT_DIR}[training,smolvla,peft]"]
)

run_quiet([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)

for module_name in list(sys.modules):
    if (
        module_name == "lerobot"
        or module_name.startswith("lerobot.")
        or module_name == "torchao"
        or module_name.startswith("torchao.")
    ):
        del sys.modules[module_name]

sys.path = [item for item in sys.path if item not in {str(LEROBOT_DIR), str(LEROBOT_SRC)}]
sys.path.insert(0, str(LEROBOT_SRC))
importlib.invalidate_caches()

try:
    importlib.metadata.version("torchao")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError("torchaoの削除に失敗しました。")

import lerobot
import peft

if LEROBOT_SRC.resolve() not in Path(lerobot.__file__).resolve().parents:
    raise RuntimeError("LeRobotの読込先が正しくありません。")

# Deviation from the Colab notebook: Colab's preinstalled torch/torchcodec
# are already a matched pair, so the notebook never (re)installs torch.
# Here, `pip install -e lerobot[...]` would otherwise resolve torchcodec to
# whatever satisfies lerobot's >=0.3,<0.12 range with no regard for ABI
# compatibility with the running torch build. torchcodec's compiled core
# library is tied 1:1 to a torch minor version, so torch==2.9.1 /
# torchvision==0.24.1 / torchcodec==0.9.1 (a verified-working pair -- see
# venv provisioning) were pre-installed into this venv *before* this
# script started, and pip's default "only-if-needed" upgrade strategy
# leaves an already-satisfying torch/torchcodec alone. We only assert that
# here rather than reinstalling, since swapping torch's compiled
# extensions inside a process that already did `import torch` is unsafe
# (the old .so stays mapped; pip installing over it does not hot-swap it).
if importlib.metadata.version("torch").split("+")[0] != "2.9.1":
    raise RuntimeError(
        f"torch was not the pre-pinned 2.9.1 (got {importlib.metadata.version('torch')}); "
        "lerobot's install step must have upgraded it unexpectedly."
    )
if importlib.metadata.version("torchcodec") != "0.9.1":
    raise RuntimeError(
        f"torchcodec was not the pre-pinned 0.9.1 (got {importlib.metadata.version('torchcodec')}); "
        "lerobot's install step must have upgraded it unexpectedly."
    )

print("LeRobot ready.")

# ---------------------------------------------------------------------------
# 4. Config
# ---------------------------------------------------------------------------
phase("4. Config")

BASE_MODEL_REPO = "lerobot/smolvla_libero_plus"
BASE_MODEL_REVISION = "7bb70aa5bc92b82c9239142775d3a173103567ff"

VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

DATASET_REPO = "lerobot/libero_plus"
DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"

TASK_NAMES = [
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "open the middle drawer of the cabinet",
    "open the top drawer and put the bowl inside",
    "pick up the alphabet soup and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the book and place it in the back compartment of the caddy",
    "pick up the butter and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the orange juice and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "push the plate to the front of the stove",
    "put both moka pots on the stove",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the bowl on the plate",
    "put the bowl on the stove",
    "put the bowl on top of the cabinet",
    "put the cream cheese in the bowl",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put the wine bottle on the rack",
    "put the wine bottle on top of the cabinet",
    "put the yellow and white mug in the microwave and close it",
    "turn on the stove",
    "turn on the stove and put the moka pot on it",
]

# Track 1 public tasks are the primary optimization target.  Keep broad
# coverage for hidden-task generalization, but expose the four relevant base
# skills substantially more often than the other 36 instructions.
TRAIN_EPISODES_PER_TASK = 4
TARGET_EPISODES_PER_TASK = 24
TRACK1_TARGET_TASKS = {
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the tomato sauce and place it in the basket",
    "pick up the milk and place it in the basket",
    "put the bowl on the stove",
}

STEPS = 8000
LOG_FREQ = 100
BATCH_SIZE = 1
LEARNING_RATE = 3e-4
FINAL_LEARNING_RATE = 3e-5
WARMUP_STEPS = 100
LORA_R = 16
LORA_ALPHA = 16
SEED = 42

EVAL_EPISODES_PER_TASK = 3
EVAL_SEED = 2026
EVAL_OBSERVATION_SIZE = 128

# Exact public Track 1 variants from compe/t1/T1_TASKS.csv.  Evaluation on a
# merely representative variant hid the actual objective (which was 0/32).
TRACK1_PUBLIC_EVAL_TASKS = {
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": (
        "libero_spatial", 80
    ),
    "pick up the tomato sauce and place it in the basket": ("libero_object", 241),
    "pick up the milk and place it in the basket": ("libero_object", 2408),
    "put the bowl on the stove": ("libero_goal", 2467),
}

OUTPUT_DIR = Path("/content/outputs/smolvla_libero_plus_track1_focused_lora")
MERGED_MODEL_DIR = Path("/content/smolvla_libero_plus_track1_focused_lora_merged")
BASELINE_MODEL_DIR = Path("/content/smolvla_libero_plus_track1_focused_baseline")

BASE_EVAL_DIR = Path("/content/eval/track1_focused_base")
FINETUNED_EVAL_DIR = Path("/content/eval/track1_focused_lora")
COMPARISON_CSV_PATH = Path("/content/track1_focused_lora_comparison.csv")
MERGED_ZIP_PATH = Path("/content/smolvla_libero_plus_track1_focused_lora_merged.zip")

MIXED_PRECISION = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

# ---------------------------------------------------------------------------
# 5. HF download helpers
# ---------------------------------------------------------------------------
import random
import time
from collections.abc import Callable
from typing import TypeVar

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

T = TypeVar("T")


def run_hf_with_retry(operation: Callable[[], T]) -> T:
    last_error: BaseException | None = None

    for attempt in range(6):
        try:
            return operation()
        except (HfHubHTTPError, httpx.HTTPStatusError) as error:
            last_error = error
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)

            if status != 429 and "429" not in str(error):
                raise

            if attempt == 5:
                break

            headers = getattr(response, "headers", {}) or {}
            try:
                delay = float(headers.get("Retry-After", 15)) + 1
            except (TypeError, ValueError):
                delay = min(120, 15 * (2**attempt) + random.random())

            time.sleep(delay)

    raise RuntimeError("Hugging Faceからの取得に失敗しました。") from last_error


def cached_or_downloaded_snapshot(
    repo_id: str,
    revision: str,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> Path:
    try:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=False,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_files_only=True,
            )
        )
    except (LocalEntryNotFoundError, FileNotFoundError):
        return Path(
            run_hf_with_retry(
                lambda: snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    token=False,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    max_workers=1,
                )
            )
        )


# ---------------------------------------------------------------------------
# 6. Select multisuite training data
# ---------------------------------------------------------------------------
phase("6. Select multisuite training episodes")

import re as _re
from collections import defaultdict

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


def normalize_task_name(value: str) -> str:
    value = value.lower().replace("_", " ")
    value = _re.sub(r"[^a-z0-9 ]+", " ", value)
    return _re.sub(r"\s+", " ", value).strip()


def task_name_from_cell(value) -> str:
    if isinstance(value, str):
        return value

    try:
        if len(value) > 0:
            return str(value[0])
    except TypeError:
        pass

    return str(value)


def choose_evenly_spaced(episode_indices: list[int], count: int) -> list[int]:
    positions = [round(index * (len(episode_indices) - 1) / (count - 1)) for index in range(count)]
    return [episode_indices[position] for position in positions]


dataset_metadata = run_hf_with_retry(lambda: LeRobotDatasetMetadata(DATASET_REPO, revision=DATASET_REVISION))

task_to_episodes: dict[str, list[int]] = defaultdict(list)

for episode_index, task_cell in enumerate(dataset_metadata.episodes["tasks"]):
    task_to_episodes[task_name_from_cell(task_cell)].append(int(episode_index))

available_by_normalized = {normalize_task_name(task_name): task_name for task_name in task_to_episodes}

selected_by_task: dict[str, list[int]] = {}

for task_name in TASK_NAMES:
    actual_task = available_by_normalized.get(normalize_task_name(task_name))

    if actual_task is None:
        raise RuntimeError(f"Training task not found: {task_name}")

    episode_count = (
        TARGET_EPISODES_PER_TASK if task_name in TRACK1_TARGET_TASKS else TRAIN_EPISODES_PER_TASK
    )
    available_episodes = task_to_episodes[actual_task]
    if len(available_episodes) < episode_count:
        raise RuntimeError(
            f"Not enough episodes for {task_name}: requested {episode_count}, "
            f"available {len(available_episodes)}"
        )
    selected_by_task[actual_task] = choose_evenly_spaced(available_episodes, episode_count)

EPISODE_INDICES = sorted(
    episode_index for episode_indices in selected_by_task.values() for episode_index in episode_indices
)

expected_episode_count = (
    (len(TASK_NAMES) - len(TRACK1_TARGET_TASKS)) * TRAIN_EPISODES_PER_TASK
    + len(TRACK1_TARGET_TASKS) * TARGET_EPISODES_PER_TASK
)
if len(EPISODE_INDICES) != expected_episode_count:
    raise RuntimeError("Episode selection failed.")

print(
    f"Training data: {len(TASK_NAMES)} tasks, {len(TRACK1_TARGET_TASKS)} Track 1 targets "
    f"oversampled to {TARGET_EPISODES_PER_TASK} episodes = {expected_episode_count} episodes"
)

# ---------------------------------------------------------------------------
# 7. Base weights
# ---------------------------------------------------------------------------
phase("7. Download base weights")

BASE_MODEL_LOCAL = cached_or_downloaded_snapshot(
    BASE_MODEL_REPO,
    BASE_MODEL_REVISION,
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_preprocessor*.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor*.safetensors",
    ],
    ignore_patterns=["README.md", "eval/**"],
)

if not (BASE_MODEL_LOCAL / "model.safetensors").is_file():
    raise FileNotFoundError("Base model not found.")

print("Base model ready.")

# ---------------------------------------------------------------------------
# 8. LoRA training
# ---------------------------------------------------------------------------
phase("8. LoRA training")

from collections import deque

episodes_json = "[" + ",".join(map(str, EPISODE_INDICES)) + "]"

command = [
    str(Path(sys.executable).parent / "lerobot-train"),
    f"--policy.path={BASE_MODEL_LOCAL}",
    f"--policy.vlm_model_name={VLM_REPO}",
    "--policy.push_to_hub=false",
    "--policy.repo_id=null",
    "--policy.input_features=null",
    "--policy.output_features=null",
    "--policy.empty_cameras=0",
    "--policy.freeze_vision_encoder=true",
    "--policy.train_expert_only=true",
    f"--policy.optimizer_lr={LEARNING_RATE}",
    f"--policy.scheduler_decay_lr={FINAL_LEARNING_RATE}",
    f"--policy.scheduler_warmup_steps={WARMUP_STEPS}",
    f"--policy.scheduler_decay_steps={STEPS}",
    f"--dataset.repo_id={DATASET_REPO}",
    f"--dataset.revision={DATASET_REVISION}",
    f"--dataset.episodes={episodes_json}",
    # Track 1 changes illumination and background materials.  LeRobot's
    # built-in transforms cover brightness/contrast/saturation/hue/sharpness
    # and small affine shifts while preserving the action labels.
    "--dataset.image_transforms.enable=true",
    "--dataset.image_transforms.max_num_transforms=4",
    "--dataset.image_transforms.random_order=true",
    "--dataset.use_imagenet_stats=false",
    "--dataset.video_backend=torchcodec",
    f"--output_dir={OUTPUT_DIR}",
    "--job_name=smolvla_libero_plus_multisuite_lora",
    f"--steps={STEPS}",
    f"--batch_size={BATCH_SIZE}",
    "--num_workers=0",
    "--persistent_workers=false",
    "--env_eval_freq=0",
    "--eval_steps=0",
    f"--seed={SEED}",
    "--save_checkpoint=true",
    f"--save_freq={STEPS}",
    "--save_checkpoint_to_hub=false",
    f"--log_freq={LOG_FREQ}",
    "--wandb.enable=false",
    "--peft.method_type=LORA",
    f"--peft.r={LORA_R}",
    f"--peft.lora_alpha={LORA_ALPHA}",
]

training_env = os.environ.copy()
training_env["PYTHONPATH"] = str(LEROBOT_SRC) + os.pathsep + training_env.get("PYTHONPATH", "")
training_env["ACCELERATE_MIXED_PRECISION"] = MIXED_PRECISION
training_env["PYTHONUNBUFFERED"] = "1"
training_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
training_env["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
training_env["HF_HUB_VERBOSITY"] = "error"
training_env["TQDM_DISABLE"] = "1"
training_env["PYTHONWARNINGS"] = "ignore"

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

print("Preparing data and starting training...", flush=True)

process = subprocess.Popen(
    command,
    cwd=LEROBOT_DIR,
    env=training_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

recent_lines: deque[str] = deque(maxlen=80)
report_step = LOG_FREQ

assert process.stdout is not None

for raw_line in process.stdout:
    line = raw_line.replace("\r", "").strip()

    if not line:
        continue

    recent_lines.append(line)

    if "step:" in line and "loss:" in line:
        loss_match = _re.search(r"loss:([0-9.eE+-]+)", line)
        lr_match = _re.search(r"lr:([0-9.eE+-]+)", line)

        loss = loss_match.group(1) if loss_match else "n/a"
        lr = lr_match.group(1) if lr_match else "n/a"

        print(f"step {report_step:4d}/{STEPS}  loss={loss}  lr={lr}", flush=True)
        report_step += LOG_FREQ

return_code = process.wait()

if return_code != 0:
    print("\n".join(recent_lines))
    raise RuntimeError(f"Training failed: {return_code}")

print("Training complete.")

# ---------------------------------------------------------------------------
# 9. Merge LoRA
# ---------------------------------------------------------------------------
phase("9. Merge LoRA into full model")

import contextlib
import gc
import io
import json as _json

from peft import PeftModel
from safetensors import safe_open
from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

checkpoint_dir = OUTPUT_DIR / "checkpoints" / f"{STEPS:06d}" / "pretrained_model"

if not (checkpoint_dir / "adapter_model.safetensors").is_file():
    raise FileNotFoundError("Final adapter not found.")

gc.collect()
torch.cuda.empty_cache()

merge_config = PreTrainedConfig.from_pretrained(checkpoint_dir)
merge_config.device = "cpu"
merge_config.pretrained_path = BASE_MODEL_LOCAL
merge_config.use_peft = False

quiet_output = io.StringIO()

with contextlib.redirect_stdout(quiet_output), contextlib.redirect_stderr(quiet_output):
    base_policy = SmolVLAPolicy.from_pretrained(BASE_MODEL_LOCAL, config=merge_config, strict=False)

    peft_policy = PeftModel.from_pretrained(base_policy, checkpoint_dir, is_trainable=False, torch_device="cpu")

    merged_policy = peft_policy.merge_and_unload(safe_merge=True)

shutil.rmtree(MERGED_MODEL_DIR, ignore_errors=True)
MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

merged_policy.config.use_peft = False
merged_policy.config.pretrained_path = None
merged_policy.config.push_to_hub = False
merged_policy.config.repo_id = None
merged_policy.config.device = None
merged_policy.config.load_vlm_weights = False
merged_policy.config.vlm_model_name = VLM_REPO

merged_policy.save_pretrained(MERGED_MODEL_DIR)

for pattern in [
    "policy_preprocessor.json",
    "policy_preprocessor*.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor*.safetensors",
]:
    for source_path in checkpoint_dir.glob(pattern):
        shutil.copy2(source_path, MERGED_MODEL_DIR / source_path.name)

merged_weights_path = MERGED_MODEL_DIR / "model.safetensors"

with safe_open(merged_weights_path, framework="pt", device="cpu") as weights:
    if any("lora_" in key.lower() for key in weights.keys()):
        raise RuntimeError("LoRA parameters remain after merge.")

del peft_policy
del base_policy
del merged_policy

gc.collect()
torch.cuda.empty_cache()

print("Merged model ready.")

# ---------------------------------------------------------------------------
# 10. Baseline (pre-finetune) model, same schema/processors
# ---------------------------------------------------------------------------
phase("10. Prepare comparison baseline")

baseline_config = PreTrainedConfig.from_pretrained(MERGED_MODEL_DIR)
baseline_config.device = "cpu"
baseline_config.pretrained_path = BASE_MODEL_LOCAL
baseline_config.use_peft = False
baseline_config.load_vlm_weights = False

quiet_output = io.StringIO()

with contextlib.redirect_stdout(quiet_output), contextlib.redirect_stderr(quiet_output):
    baseline_policy = SmolVLAPolicy.from_pretrained(BASE_MODEL_LOCAL, config=baseline_config, strict=False)

shutil.rmtree(BASELINE_MODEL_DIR, ignore_errors=True)
BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

baseline_policy.config.use_peft = False
baseline_policy.config.pretrained_path = None
baseline_policy.config.push_to_hub = False
baseline_policy.config.repo_id = None
baseline_policy.config.device = None
baseline_policy.config.load_vlm_weights = False
baseline_policy.config.vlm_model_name = VLM_REPO
baseline_policy.save_pretrained(BASELINE_MODEL_DIR)

for pattern in [
    "policy_preprocessor.json",
    "policy_preprocessor*.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor*.safetensors",
]:
    for source_path in MERGED_MODEL_DIR.glob(pattern):
        shutil.copy2(source_path, BASELINE_MODEL_DIR / source_path.name)

del baseline_policy
gc.collect()
torch.cuda.empty_cache()

print("Baseline ready.")

# ---------------------------------------------------------------------------
# 11. LIBERO-plus eval environment (EGL, GPU-accelerated rendering)
# ---------------------------------------------------------------------------
phase("11. LIBERO-plus eval environment")

from huggingface_hub import hf_hub_download

LIBERO_PLUS_SHA = "4976dc3"
LIBERO_PLUS_DIR = Path("/content/LIBERO-plus")
LIBERO_PLUS_PACKAGE_ROOT = LIBERO_PLUS_DIR / "libero" / "libero"
LIBERO_PLUS_ASSETS_DIR = LIBERO_PLUS_PACKAGE_ROOT / "assets"

os.environ["MUJOCO_GL"] = "egl"

run_quiet([sys.executable, "-m", "pip", "uninstall", "-y", "hf-libero", "libero", "robosuite"], check=False)

run_quiet(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "robosuite==1.4.1",
        "bddl==1.0.1",
        "easydict==1.13",
        "mujoco==3.7.0",
        "matplotlib==3.10.8",
        "Wand==0.6.13",
        "scikit-image==0.25.2",
        "gym==0.26.2",
        # Not in the notebook: bddl==1.0.1's code imports `future` (py2/3
        # compat shim) but its packaging metadata doesn't declare it as a
        # dependency, so pip never pulls it in. Colab happens to have it
        # preinstalled already; a bare venv does not.
        "future",
    ]
)

if importlib.metadata.version("robosuite") != "1.4.1":
    raise RuntimeError("robosuite 1.4.1 is required.")

if not (LIBERO_PLUS_DIR / ".git").is_dir():
    shutil.rmtree(LIBERO_PLUS_DIR, ignore_errors=True)
    run_quiet(["git", "clone", "--quiet", "https://github.com/sylvestf/LIBERO-plus.git", str(LIBERO_PLUS_DIR)])

checkout = run_quiet(["git", "-C", str(LIBERO_PLUS_DIR), "checkout", "--quiet", LIBERO_PLUS_SHA], check=False)

if checkout.returncode != 0:
    run_quiet(["git", "-C", str(LIBERO_PLUS_DIR), "fetch", "--quiet", "--depth", "1", "origin", LIBERO_PLUS_SHA])
    run_quiet(["git", "-C", str(LIBERO_PLUS_DIR), "checkout", "--quiet", LIBERO_PLUS_SHA])

run_quiet([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(LIBERO_PLUS_DIR)])

if not LIBERO_PLUS_ASSETS_DIR.is_dir():
    assets_root = Path("/content/libero_plus_assets")
    archive_path = Path(
        run_hf_with_retry(
            lambda: hf_hub_download(
                repo_id="Sylvest/LIBERO-plus",
                repo_type="dataset",
                filename="assets.zip",
                local_dir=assets_root,
                token=False,
            )
        )
    )
    extract_dir = assets_root / "extract"

    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    run_quiet(["unzip", "-q", str(archive_path), "-d", str(extract_dir)])

    candidates = sorted(
        [path for path in extract_dir.rglob("assets") if path.is_dir()],
        key=lambda path: len(path.parts),
    )

    if not candidates:
        raise FileNotFoundError("LIBERO-plus assets not found.")

    LIBERO_PLUS_ASSETS_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(candidates[0]), str(LIBERO_PLUS_ASSETS_DIR))
    shutil.rmtree(assets_root, ignore_errors=True)

libero_config_dir = Path.home() / ".libero"
libero_config_dir.mkdir(parents=True, exist_ok=True)
(libero_config_dir / "config.yaml").write_text(
    "\n".join(
        [
            f"assets: {LIBERO_PLUS_ASSETS_DIR}",
            f"bddl_files: {LIBERO_PLUS_PACKAGE_ROOT / 'bddl_files'}",
            f"datasets: {LIBERO_PLUS_PACKAGE_ROOT.parent / 'datasets'}",
            f"init_states: {LIBERO_PLUS_PACKAGE_ROOT / 'init_files'}",
        ]
    )
    + "\n",
    encoding="utf-8",
)

eval_script = LEROBOT_SRC / "lerobot" / "scripts" / "lerobot_eval.py"
source = eval_script.read_text(encoding="utf-8")

source = source.replace("logging.info(pformat(asdict(cfg)))", "logging.debug(pformat(asdict(cfg)))", 1)
source = source.replace(
    "max_episodes_rendered = 0 if cfg.eval.recording else 10", "max_episodes_rendered = 0", 1
)
source = source.replace("disable=inside_slurm()", "disable=True")

progress_state = '_EVAL_PROGRESS = {"task_index": 0, "task_total": 0}'
if progress_state not in source:
    import_anchor = "from tqdm import trange\n"
    if import_anchor not in source:
        raise RuntimeError("Evaluation progress import anchor not found.")
    source = source.replace(import_anchor, import_anchor + "\n" + progress_state + "\n", 1)

task_loop_anchor = "        for i, (task_group, task_id, env) in enumerate(tasks):\n"
task_loop_patch = (
    task_loop_anchor
    + '            _EVAL_PROGRESS["task_index"] = i + 1\n'
    + '            _EVAL_PROGRESS["task_total"] = len(tasks)\n'
)
if '_EVAL_PROGRESS["task_index"] = i + 1' not in source:
    if task_loop_anchor not in source:
        raise RuntimeError("Evaluation task-loop anchor not found.")
    source = source.replace(task_loop_anchor, task_loop_patch, 1)

episode_loop_anchor = "    for batch_ix in progbar:\n"
episode_progress_line = (
    "        print("
    'f"EVAL_PROGRESS '
    "task={_EVAL_PROGRESS['task_index']}/"
    "{_EVAL_PROGRESS['task_total']} "
    'episode={batch_ix + 1}/{n_batches}", '
    "flush=True)\n"
)
if "EVAL_PROGRESS task=" not in source:
    if episode_loop_anchor not in source:
        raise RuntimeError("Evaluation episode-loop anchor not found.")
    source = source.replace(episode_loop_anchor, episode_loop_anchor + episode_progress_line, 1)

eval_script.write_text(source, encoding="utf-8")

libero_plus_path = str(LIBERO_PLUS_DIR)
sys.path = [item for item in sys.path if item != libero_plus_path]
sys.path.insert(0, libero_plus_path)

for module_name in list(sys.modules):
    if (
        module_name == "libero"
        or module_name.startswith("libero.")
        or module_name == "robosuite"
        or module_name.startswith("robosuite.")
    ):
        del sys.modules[module_name]

importlib.invalidate_caches()

import libero
from libero.libero import benchmark

search_paths = [Path(path).resolve() for path in getattr(libero, "__path__", [])]

if not any(
    LIBERO_PLUS_DIR.resolve() in path.parents or path == LIBERO_PLUS_DIR.resolve() for path in search_paths
):
    raise RuntimeError("LIBERO-plus fork was not loaded.")

benchmark_path = Path(benchmark.__file__).resolve()

if LIBERO_PLUS_DIR.resolve() not in benchmark_path.parents:
    raise RuntimeError("LIBERO-plus benchmark was not loaded.")

print("LIBERO-plus ready.")

LIBERO_PLUS_TASK_CLASSIFICATION = _json.loads(
    (LIBERO_PLUS_PACKAGE_ROOT / "benchmark" / "task_classification.json").read_text(encoding="utf-8")
)

# Each LIBERO-plus suite's ~2400-2600 "tasks" are NOT distinct instructions --
# they are dozens/hundreds of perturbation *variants* of the 40 base
# instructions above (language "<base instruction> table 1", "<base
# instruction> table 10", ...; category comes from task_classification.json,
# which is index-aligned 1:1 with the benchmark's own task order -- verified
# separately by diffing task.name against task_classification.json[suite][i]
# ["name"] for every task in every suite: zero mismatches). The 40 base
# instructions are also not confined to libero_spatial/object/goal -- 10 of
# them (the multi-step/long-horizon ones, e.g. "put both moka pots on the
# stove") only exist in the libero_10 suite. CATEGORY_PREFERENCE biases the
# single representative variant picked per base instruction toward the
# perturbation categories Track1 actually grades on
# (compe/t1/T1_TASKS.csv uses "Background Textures" and "Light Conditions").
CATEGORY_PREFERENCE = ["Background Textures", "Light Conditions"]

EVAL_SUITE_NAMES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
normalized_base_names = {normalize_task_name(name): name for name in TASK_NAMES}
matches_by_base_name: dict[str, list[tuple[str, int, str]]] = {name: [] for name in TASK_NAMES}

for suite_name in EVAL_SUITE_NAMES:
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    categories = [entry["category"] for entry in LIBERO_PLUS_TASK_CLASSIFICATION[suite_name]]
    if len(categories) != task_suite.n_tasks:
        raise RuntimeError(f"task_classification.json length mismatch for {suite_name}")

    for task_id in range(task_suite.n_tasks):
        # "Language Instructions" variants deliberately reword the instruction
        # (e.g. "turn on the stove and set the moka pot on it but don't
        # forget..."), so substring-matching them against the original text
        # is unreliable/accidental (a reworded moka-pot variant can spuriously
        # prefix-match the unrelated, shorter "turn on the stove" base
        # instruction). Since these aren't in CATEGORY_PREFERENCE anyway, skip
        # collecting them as match candidates entirely.
        if categories[task_id] == "Language Instructions":
            continue

        task = task_suite.get_task(task_id)
        normalized_lang = normalize_task_name(str(task.language))

        # Some base instructions are textual prefixes of others (e.g. "turn on
        # the stove" vs. "turn on the stove and put the moka pot on it"), so
        # take the *longest* matching base name rather than treating this as
        # an ambiguity error.
        matched_base = None
        for norm_base, base_name in normalized_base_names.items():
            if normalized_lang == norm_base or normalized_lang.startswith(norm_base + " "):
                if matched_base is None or len(norm_base) > len(normalize_task_name(matched_base)):
                    matched_base = base_name

        if matched_base is not None:
            matches_by_base_name[matched_base].append((suite_name, task_id, categories[task_id]))

unmatched_eval_tasks = [name for name, matches in matches_by_base_name.items() if not matches]
cross_suite_eval_tasks = {
    name: matches
    for name, matches in matches_by_base_name.items()
    if len({suite for suite, _, _ in matches}) > 1
}

if unmatched_eval_tasks or cross_suite_eval_tasks:
    details = []
    if unmatched_eval_tasks:
        details.append("unmatched=" + repr(unmatched_eval_tasks))
    if cross_suite_eval_tasks:
        details.append("cross_suite=" + repr(list(cross_suite_eval_tasks)))
    raise RuntimeError("Eval task matching failed: " + "; ".join(details))


def pick_representative_eval_task(matches: list[tuple[str, int, str]]) -> tuple[str, int, str]:
    def sort_key(match: tuple[str, int, str]) -> tuple[int, int]:
        _, task_id, category = match
        preference = (
            CATEGORY_PREFERENCE.index(category) if category in CATEGORY_PREFERENCE else len(CATEGORY_PREFERENCE)
        )
        return (preference, task_id)

    return sorted(matches, key=sort_key)[0]


EVAL_TASKS_BY_SUITE: dict[str, list[tuple[int, str]]] = {suite_name: [] for suite_name in EVAL_SUITE_NAMES}
EVAL_TASK_LABELS: dict[tuple[str, int], str] = {}

for base_name, (suite_name, task_id) in TRACK1_PUBLIC_EVAL_TASKS.items():
    matches = matches_by_base_name[base_name]
    if not any(match_suite == suite_name and match_id == task_id for match_suite, match_id, _ in matches):
        raise RuntimeError(
            f"Track 1 public task mapping is stale: {base_name!r} -> ({suite_name!r}, {task_id})"
        )
    EVAL_TASKS_BY_SUITE[suite_name].append((task_id, base_name))
    EVAL_TASK_LABELS[(suite_name, task_id)] = base_name

print(
    "Track 1 public eval tasks matched: "
    + ", ".join(
        f"{suite_name}={len(EVAL_TASKS_BY_SUITE[suite_name])}" for suite_name in EVAL_SUITE_NAMES
    )
)

# ---------------------------------------------------------------------------
# 12. Evaluate before/after
# ---------------------------------------------------------------------------
phase("12. Evaluate base vs. multisuite LoRA across all suites")

EVAL_CAMERA_MAPPING = {
    "agentview_image": "front",
    "robot0_eye_in_hand_image": "wrist",
}


def build_eval_command(
    policy_path: Path, output_dir: Path, suite_name: str, task_ids: list[int]
) -> list[str]:
    return [
        str(Path(sys.executable).parent / "lerobot-eval"),
        f"--policy.path={policy_path}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--env.type=libero",
        "--env.is_libero_plus=true",
        f"--env.task={suite_name}",
        "--env.task_ids=" + _json.dumps(task_ids, separators=(",", ":")),
        "--env.camera_name_mapping=" + _json.dumps(EVAL_CAMERA_MAPPING, separators=(",", ":")),
        f"--env.observation_height={EVAL_OBSERVATION_SIZE}",
        f"--env.observation_width={EVAL_OBSERVATION_SIZE}",
        "--env.control_mode=relative",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        f"--eval.n_episodes={EVAL_EPISODES_PER_TASK}",
        "--eval.use_async_envs=false",
        "--eval.recording=false",
        f"--seed={EVAL_SEED}",
        f"--output_dir={output_dir}",
    ]


def run_evaluation(
    policy_path: Path,
    output_dir: Path,
    label: str,
    suite_name: str,
    task_ids: list[int],
) -> dict:
    shutil.rmtree(output_dir, ignore_errors=True)

    eval_env = os.environ.copy()
    eval_env["MUJOCO_GL"] = "egl"
    eval_env["PYTHONPATH"] = (
        str(LIBERO_PLUS_DIR) + os.pathsep + str(LEROBOT_SRC) + os.pathsep + eval_env.get("PYTHONPATH", "")
    )
    eval_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    eval_env["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    eval_env["HF_HUB_VERBOSITY"] = "error"
    eval_env["TQDM_DISABLE"] = "1"
    eval_env["PYTHONWARNINGS"] = "ignore"
    eval_env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        build_eval_command(policy_path, output_dir, suite_name, task_ids),
        cwd=LEROBOT_DIR,
        env=eval_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    recent_lines: deque[str] = deque(maxlen=120)

    progress_pattern = _re.compile(r"^EVAL_PROGRESS task=(\d+)/(\d+) episode=(\d+)/(\d+)$")

    assert process.stdout is not None

    for raw_line in process.stdout:
        line = raw_line.replace("\r", "").strip()

        if not line:
            continue

        recent_lines.append(line)
        match = progress_pattern.match(line)

        if match:
            task_index, task_total, episode_index, episode_total = match.groups()
            print(
                f"{label:<35} | task {task_index}/{task_total} | episode {episode_index}/{episode_total}",
                flush=True,
            )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError("\n".join(recent_lines))

    result_path = output_dir / "eval_info.json"

    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    return _json.loads(result_path.read_text(encoding="utf-8"))


BASE_EVAL_INFO_BY_SUITE: dict[str, dict] = {}
FINETUNED_EVAL_INFO_BY_SUITE: dict[str, dict] = {}

for suite_name in EVAL_SUITE_NAMES:
    task_ids = sorted(task_id for task_id, _ in EVAL_TASKS_BY_SUITE[suite_name])

    if not task_ids:
        continue

    BASE_EVAL_INFO_BY_SUITE[suite_name] = run_evaluation(
        BASELINE_MODEL_DIR,
        BASE_EVAL_DIR / suite_name,
        f"Base model [{suite_name}]",
        suite_name,
        task_ids,
    )
    FINETUNED_EVAL_INFO_BY_SUITE[suite_name] = run_evaluation(
        MERGED_MODEL_DIR,
        FINETUNED_EVAL_DIR / suite_name,
        f"Multisuite LoRA [{suite_name}]",
        suite_name,
        task_ids,
    )

print("Evaluation complete.")

# ---------------------------------------------------------------------------
# 13. Compare success rates
# ---------------------------------------------------------------------------
phase("13. Compare success rates")

import pandas as pd

try:
    from IPython.display import display
except ModuleNotFoundError:
    # Not in the notebook: Colab always has IPython preinstalled. A bare
    # venv does not, and it isn't needed outside a notebook frontend --
    # display() there just falls back to a plain print of the object.
    def display(obj) -> None:  # noqa: A001
        print(obj)


def per_task_success(eval_info: dict) -> dict[int, float]:
    result: dict[int, float] = {}

    for task_info in eval_info["per_task"]:
        task_id = int(task_info["task_id"])
        successes = task_info["metrics"]["successes"]
        result[task_id] = 100.0 * sum(bool(value) for value in successes) / len(successes)

    return result


rows = []

for suite_name in EVAL_SUITE_NAMES:
    matched_task_ids = sorted(task_id for task_id, _ in EVAL_TASKS_BY_SUITE[suite_name])

    if not matched_task_ids:
        continue

    base_eval_info = BASE_EVAL_INFO_BY_SUITE[suite_name]
    finetuned_eval_info = FINETUNED_EVAL_INFO_BY_SUITE[suite_name]
    base_per_task = per_task_success(base_eval_info)
    finetuned_per_task = per_task_success(finetuned_eval_info)

    for task_id in matched_task_ids:
        base_score = base_per_task[task_id]
        finetuned_score = finetuned_per_task[task_id]

        rows.append(
            {
                "Suite": suite_name,
                "Task ID": task_id,
                "Task": EVAL_TASK_LABELS[(suite_name, task_id)],
                "Base (%)": base_score,
                "Multisuite LoRA (%)": finetuned_score,
                "Delta (pp)": finetuned_score - base_score,
            }
        )

    base_suite_overall = float(base_eval_info["overall"]["pc_success"])
    finetuned_suite_overall = float(finetuned_eval_info["overall"]["pc_success"])
    rows.append(
        {
            "Suite": suite_name,
            "Task ID": "Overall",
            "Task": f"{suite_name} (n={len(matched_task_ids)} tasks)",
            "Base (%)": base_suite_overall,
            "Multisuite LoRA (%)": finetuned_suite_overall,
            "Delta (pp)": finetuned_suite_overall - base_suite_overall,
        }
    )


def aggregate_success(eval_info_by_suite: dict[str, dict]) -> float:
    success_count = 0
    episode_count = 0

    for eval_info in eval_info_by_suite.values():
        for task_info in eval_info["per_task"]:
            successes = task_info["metrics"]["successes"]
            success_count += sum(bool(value) for value in successes)
            episode_count += len(successes)

    if episode_count == 0:
        raise RuntimeError("Evaluation produced no episodes.")

    return 100.0 * success_count / episode_count


base_overall = aggregate_success(BASE_EVAL_INFO_BY_SUITE)
finetuned_overall = aggregate_success(FINETUNED_EVAL_INFO_BY_SUITE)

rows.append(
    {
        "Suite": "ALL",
        "Task ID": "Overall",
        "Task": f"All suites (n={len(EVAL_TASK_LABELS)} tasks)",
        "Base (%)": base_overall,
        "Multisuite LoRA (%)": finetuned_overall,
        "Delta (pp)": finetuned_overall - base_overall,
    }
)

comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)

display(comparison_df.round(1))

print(f"Overall: {base_overall:.1f}% -> {finetuned_overall:.1f}% ({finetuned_overall - base_overall:+.1f} pp)")

# ---------------------------------------------------------------------------
# 14. Done (local: no browser download, just print result paths)
# ---------------------------------------------------------------------------
phase("14. Done")

print(f"Merged model dir: {MERGED_MODEL_DIR}")
print(f"Baseline model dir: {BASELINE_MODEL_DIR}")
print(f"Comparison CSV: {COMPARISON_CSV_PATH}")
print("Multisuite LoRA run complete.")
