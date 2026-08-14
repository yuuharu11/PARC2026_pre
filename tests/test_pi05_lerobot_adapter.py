import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "submission_template" / "pi05_lerobot_policy.py"
SPEC = importlib.util.spec_from_file_location("pi05_lerobot_policy", MODULE_PATH)
pi05 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pi05)


def test_axis_angle_identity():
    np.testing.assert_array_equal(pi05._axis_angle([0.0, 0.0, 0.0, 1.0]), np.zeros(3))


def test_state_layout():
    obs = {
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 1.0, 0.0]),
        "robot0_gripper_qpos": np.array([0.1, 0.2]),
    }
    np.testing.assert_allclose(
        pi05._build_state(obs),
        [1.0, 2.0, 3.0, 0.0, 0.0, np.pi, 0.1, 0.2],
        rtol=1e-6,
    )


def test_image_tensor_preserves_orientation_and_scales():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[0, 0] = [255, 128, 0]
    tensor = pi05._image_tensor(image)
    assert tuple(tensor.shape) == (3, 4, 5)
    np.testing.assert_allclose(tensor[:, 0, 0].numpy(), [1.0, 128 / 255, 0.0])


def test_image_tensor_can_flip_for_openpi_comparison():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[0, 0] = [255, 128, 0]
    tensor = pi05._image_tensor(image, flip=True)
    np.testing.assert_allclose(tensor[:, -1, -1].numpy(), [1.0, 128 / 255, 0.0])
