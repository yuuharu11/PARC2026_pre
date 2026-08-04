"""ポリシーサーバー（提出用テンプレート）

このファイルを編集して、自分のモデルを組み込んでください。
編集が必要なのは MyPolicy クラスの中身だけです。
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可です。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

import argparse
from abc import ABC, abstractmethod

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。以下のキーが含まれる:
                - "agentview_image": (128, 128, 3) uint8
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8
                - "robot0_joint_pos": (7,) float
                - "robot0_eef_pos": (3,) float
                - "robot0_eef_quat": (4,) float
                - "robot0_gripper_qpos": (2,) float

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


class MyPolicy(BasePolicy):
    """Pretrained SmolVLA policy for LIBERO-plus."""

    def __init__(self):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = "lerobot/smolvla_libero_plus"

        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        config.device = str(self.device)
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint, config=config)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
            },
        )
        self.instruction = ""

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        torch = self._torch

        def image_tensor(image: np.ndarray):
            return (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .contiguous()
                .to(dtype=torch.float32)
                .div_(255.0)
            )

        # State layout (8-dim): the checkpoint's config.json claims shape (6,), but
        # the actual saved normalizer stats (policy_preprocessor_step_5_normalizer_
        # processor.safetensors) are (8,) with value ranges that match eef_pos(3)
        # + eef_euler_xyz(3, spanning ~[-pi, pi]) + gripper_qpos(2, near-symmetric
        # +/- pair) -- confirmed empirically against those stats, not from docs.
        state = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                Rotation.from_quat(obs["robot0_eef_quat"]).as_euler("xyz").astype(np.float32),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            ]
        )

        batch = {
            "observation.state": torch.from_numpy(state),
            "observation.images.camera1": image_tensor(obs["agentview_image"]),
            "observation.images.camera2": image_tensor(obs["robot0_eye_in_hand_image"]),
            "observation.images.camera3": torch.zeros((3, 256, 256), dtype=torch.float32),
            "observation.images.empty_camera_0": torch.zeros(
                (3, 480, 640), dtype=torch.float32
            ),
            "observation.images.empty_camera_1": torch.zeros(
                (3, 480, 640), dtype=torch.float32
            ),
            "task": self.instruction,
        }
        batch = self.preprocessor(batch)

        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)

        return action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def reset(self, instruction: str = "") -> None:
        self.instruction = instruction
        self.policy.reset()


# ============================================================
# 以下は変更不可
# ============================================================


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
