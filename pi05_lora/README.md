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
