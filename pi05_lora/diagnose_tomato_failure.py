"""Run a single tomato-sauce episode against a running policy server, saving
periodic camera frames and tracking gripper-to-object distance, to diagnose
*how* the policy is failing (never approaches vs. grasps-then-drops vs.
picks the wrong object, etc.) rather than guessing from the success rate
alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import EvalConfig, PerturbationConfig
from pipeline.environment import EnvironmentManager
from pipeline.remote_policy import RemotePolicyClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8010")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--frame-interval", type=int, default=20)
    parser.add_argument("--episode-id", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eval_config = EvalConfig()
    env_manager = EnvironmentManager(eval_config)
    tasks = env_manager.get_task_infos("libero_t1")
    task_info = next(t for t in tasks if "tomato_sauce" in t.name and "basket" in t.name)
    print(f"task: {task_info.name} | instruction: {task_info.language!r}")

    env = env_manager.create_env(task_info)
    init_state = np.asarray(task_info.init_states[args.episode_id])

    policy = RemotePolicyClient(args.server_url)
    policy.wait_for_server()

    env.reset()
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    action_dim = env.robots[0].action_dim
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(action_dim))

    obj_keys = [k for k in obs if k.endswith("_pos") and not k.startswith("robot0")]
    print("object keys in obs:", obj_keys)
    tomato_key = next((k for k in obj_keys if "tomato" in k.lower()), None)
    print("tracking tomato object key:", tomato_key)

    policy.reset(instruction=task_info.language, seed=eval_config.n_eval_episodes + args.episode_id)

    Image.fromarray(obs["agentview_image"]).save(args.out_dir / "frame_0000_reset.png")

    for step in range(args.max_steps):
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)

        if tomato_key:
            eef = np.asarray(obs["robot0_eef_pos"])
            tom = np.asarray(obs[tomato_key])
            dist = float(np.linalg.norm(eef - tom))
        else:
            dist = float("nan")
        gripper = np.asarray(obs["robot0_gripper_qpos"])

        if step % args.frame_interval == 0 or done:
            Image.fromarray(obs["agentview_image"]).save(
                args.out_dir / f"frame_{step:04d}.png"
            )
            print(
                f"step={step:4d} eef-to-tomato-dist={dist:.4f} "
                f"gripper_qpos={gripper} done={done}"
            )

        if done:
            break

    print(f"final: total_steps={step + 1}, done={done}")


if __name__ == "__main__":
    main()
