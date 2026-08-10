# Single-model generalization investigation

Date: 2026-08-09

## Decision

The best *single, public, multi-suite* model to test next is
**pi0.5-LIBERO**. It is preferable to the currently integrated VLANeXt
checkpoints for this competition because the official pi0.5 checkpoint is one
checkpoint trained across the LIBERO suites. The public VLANeXt release has
four separate suite checkpoints; selecting one from the task name is not a
single general policy and is likely to fail hidden tasks.

This investigation did **not** establish a competition-test score above 60%.
Do not report the standard-LIBERO numbers below as a PARC or LIBERO-Plus score.

## Hard constraints

- one checkpoint and one learned policy for every task;
- submission archive/model budget: 20 GiB;
- offline startup, including model loading: at most 120 seconds;
- every `/reset` and `/act`: at most 10 seconds;
- 7-DoF LIBERO action output and two-camera input;
- robustness to unseen tasks and visual perturbations matters more than
  memorizing the four public tasks.

## Candidate audit

| Candidate | Single multi-suite public checkpoint | Generalization evidence | Packaging | Verdict |
|---|---:|---|---|---|
| pi0.5-LIBERO | yes | 96.85% mean on standard LIBERO (98.8 spatial, 98.2 object, 98.0 goal, 92.4 long) | public JAX checkpoint is about 12.44 GB before dependencies; official inference requirement is >8 GB VRAM | first choice, but PARC score and latency still unmeasured |
| VLANeXt 2.5B | no, not in the public release inspected | paper reports 83.9% on LIBERO-Plus | each suite checkpoint is 16.4 GB; four total 65.8 GB | excellent research result, unsuitable as a public single-checkpoint submission today |
| GR00T N1.7 3B LIBERO | no, release is split into suite directories | no directly comparable official LIBERO-Plus result found | approximately 6 GB per checkpoint; official docs require about 16 GB VRAM | not selected |
| SmolVLA 450M currently bundled | yes | competition public run was 0/16 on tasks delegated to the model | 906.7 MB model weights | fast and compact, but existing evidence is far below 60% |
| OpenVLA-OFT 7B | yes | official LIBERO-Plus leaderboard exceeds 60% | model/dependency and latency risk are materially higher | evidence is strong, but conflicts with the requested ~3B/competition constraints |

The newer research claims (for example TFP) were not selected because a
reproducible public checkpoint under the same evaluation protocol was not
verified. A paper number alone is not treated as a test result.

## What was actually tested here

### VLANeXt

- Downloaded and compacted the official object-suite checkpoint.
- Verified 3,051,084,807 parameters and successfully loaded it with zero
  missing or unexpected tensors.
- CPU load time was 86.21 seconds after removing training-only state.
- An A100-PCIE-40GB became available outside the normal command sandbox. With
  five diffusion steps, cold load was 94.54 seconds, the first action chunk was
  1.48 seconds, a later chunk was 0.17 seconds, and peak reserved VRAM was
  6.31 GiB. These pass the latency constraints on that GPU.
- No rollout score has yet been produced, and the tested checkpoint remains
  Object-suite-specific rather than a general mixed policy.
- See `VLANEXT_EXPERIMENT.md` for exact integration details.

### pi0.5-LIBERO

- Verified the official public GCS checkpoint object list rather than relying
  on an estimated parameter count.
- The weight payload is approximately 12.44 GB and therefore fits inside the
  20 GiB model ceiling in isolation.
- An A100-PCIE-40GB was available outside the normal command sandbox. The
  official JAX checkpoint loaded in 8.43 seconds. Its first JIT inference took
  12.61 seconds, so the PARC adapter deliberately compiles before `/health` is
  exposed. Adapter construction plus warm-up took 26.02 seconds; subsequent
  action-chunk inference was about 0.10 seconds. This passes the 120-second
  startup and 10-second request limits.
- The full public Track 1 test (four tasks x eight initial states) scored
  **26/32 = 81.25%**, using one unmodified pi0.5-LIBERO checkpoint and no
  scripted controller. Per-task results were 7/8 drawer bowl-to-plate, 3/8
  tomato-to-basket, 8/8 perturbed-light milk-to-basket, and 8/8
  perturbed-light bowl-to-stove. This passes the predeclared 20/32 acceptance
  threshold. The exact result is recorded in
  `results/pi05_track1_8ep/server_8014.json`.

### Existing generalized state machine

The old public result is 50% (16/32), but all successes came from scripted
tomato/milk transfers; SmolVLA contributed 0/16. This is not a single-model
result. An attempted bowl-to-stove rerun did not produce a score because the
policy-server process was terminated by the interactive execution session
before the evaluator connected. It must not be counted as either success or
failure.

## Next experiment (required to answer “over 60%?”)

1. Run on a CUDA machine with at least 16 GB VRAM (24 GB preferred).
2. Convert or package pi0.5-LIBERO for offline loading; include code,
   tokenizer/assets and dependencies in the 20 GiB archive check.
3. Implement the PARC observation adapter using the official LIBERO input and
   output transforms. Preserve the model's action chunking.
4. Measure cold startup and the slowest chunk inference, not just averages.
   Reject the build if startup exceeds 120 seconds or any request exceeds 10
   seconds.
5. ~~Evaluate all four public tasks with at least 8 seeds (32 episodes).~~
   Completed at 26/32 with no request timeout. Next, run held-out LIBERO-Plus
   perturbations before treating it as a genuinely general solution.

The 20/32 threshold is deliberate: 19/32 is 59.375%, so it does not exceed
60%.

## Sources

- Physical Intelligence openpi repository and LIBERO README (official pi0.5
  checkpoint, requirements and standard-LIBERO results)
- Official LIBERO-Plus repository/leaderboard
- Official VLANeXt repository, checkpoint repository and paper
- Official NVIDIA GR00T N1.7 LIBERO checkpoint repository
- Local competition README, evaluator configuration and recorded JSON results
