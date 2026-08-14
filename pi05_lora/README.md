# pi0.5 LIBERO LoRA

The baseline submission remains untouched.  Training data and checkpoints are
kept outside the repository:

- dataset: `/work/PARC2026_data/lerobot/physical-intelligence/libero`
- checkpoints: `/work/PARC2026_training/checkpoints/pi05_libero_lora`

Run the 20-step smoke test with the openpi environment and an A100:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
/tmp/openpi/.venv/bin/python pi05_lora/train_pi05_lora.py \
  --steps 20 --batch-size 8 --save-interval 10 --overwrite
```

If the full 40-task download is still in progress, make a no-copy subset from
the available contiguous prefix and use it only for a training smoke test:

```bash
/tmp/openpi/.venv/bin/python pi05_lora/prepare_smoke_dataset.py
/tmp/openpi/.venv/bin/python pi05_lora/train_pi05_lora.py \
  --dataset-repo-id physical-intelligence/libero-smoke \
  --steps 20 --batch-size 8 --save-interval 10 --overwrite
```

Restore the resulting LoRA-shaped checkpoint and compile one inference:

```bash
/tmp/openpi/.venv/bin/python pi05_lora/verify_checkpoint.py
```

The submission adapter auto-detects a LoRA checkpoint when
`training_manifest.json` is beside the numbered checkpoint (or copied into its
root). It can also be selected explicitly with `PI05_VARIANT=lora`.

The model uses the official pi0.5-LIBERO checkpoint as its initial state,
Gemma 2B LoRA rank 16, and action-expert LoRA rank 32.  Every non-LoRA
parameter is frozen: about 50.0M of 3.40B parameters (1.47%) are trainable.
EMA is disabled.

## LIBERO-Plus dataset pipeline

`lerobot/libero_plus` (HF Hub, v3.0 schema, fps=20) is not directly readable
by openpi's pinned old-lerobot. Two stages convert it into an openpi-compatible
flat-key dataset:

- `stage1_decode_libero_plus.py` (main venv): decode selected episodes into
  per-episode `.npz` files.
- `stage2_build_openpi_dataset.py` (openpi venv): rebuild those npz files into
  an openpi-format LeRobotDataset (fps=20, flat `image`/`wrist_image` keys).
  Supports `--append` to extend an existing dataset.
- `merge_shard_datasets.py`: filesystem-level merge of independently-built
  shard datasets into a master, without going through LeRobotDataset's slow
  per-frame API.
- `extract_task_dataset.py`: pull all episodes of one task out into a
  standalone single-task dataset (used to isolate a task from multi-task LoRA
  interference for diagnostics).
- `oversample_task.py`: duplicate one task's episodes N-1 extra times in
  place, to bias a multi-task LoRA's training-batch composition toward that
  task using only real data (no synthetic augmentation).

All three of the above that copy parquet data (`merge_shard_datasets.py`,
`extract_task_dataset.py`, `oversample_task.py`) use `pyarrow` directly rather
than pandas' `read_parquet`/`to_parquet` convenience methods, which silently
drop the parquet schema's `huggingface` metadata key that tells HF `datasets`
to auto-decode image columns. Losing it makes every image column load as a
raw `{"bytes", "path"}` dict instead of a tensor, and training crashes deep in
`hf_transform_to_torch` with "Could not infer dtype of dict".

## Tomato-sauce augmentation and diagnostics

The tomato-sauce Track1 task persistently underperformed the other three
graded tasks across every tested configuration. `diagnose_tomato_failure.py`
and `scan_tomato_brightness.py` were used to root-cause it: the eval scene's
texture perturbation reuses a PBR GLOSS channel as the table's color texture,
producing a near-black scene (~30/255 mean brightness) with no matching or
even closely-matching example in the available training demonstrations
(darkest real episode found was ~45/255).

`texture_perturbation.py` (a `TexturePerturbation` dataclass) mimics this and
two other perturbation modes found by inspecting LIBERO-Plus's actual texture
assets (NRM channel -> blue/purple tint, REFL/AO/DISP -> desaturation) as a
training-time-only augmentation via `repack_transforms` (never applied at
inference). `augmented_data_config.py` wires it into training via `--augment`
on `train_pi05_lora.py`, with optional task-conditional boosting
(`--boosted-prob`, `--boosted-dark-weight`) for the tomato-sauce task
specifically.

**Result:** synthetic augmentation, at every tested strength, underperformed
training on real data alone -- confirmed both in the full 4-task mix and in
an isolated tomato-only ablation (43.8% with no augmentation vs. 31.2% at
moderate augmentation and 6.2% at strong augmentation, all n=32). The
multi-task LoRA's shared capacity across four tasks was the larger factor:
oversampling tomato-sauce's real episodes 3x via `oversample_task.py` (no
synthetic augmentation) raised Track1 overall from 75.0% to 76.6% and
tomato-sauce specifically from 25.0% to 31.2% (n=16/task), and is the
configuration used for the `pi05_submission_0811.zip` candidate.
