import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "submission_template" / "pi05_policy.py"
SPEC = importlib.util.spec_from_file_location("pi05_policy", MODULE_PATH)
pi05 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pi05)


def test_axis_angle_identity():
    np.testing.assert_array_equal(pi05._axis_angle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3))


def test_axis_angle_half_turn():
    value = pi05._axis_angle(np.array([0.0, 0.0, 1.0, 0.0]))
    np.testing.assert_allclose(value, [0.0, 0.0, np.pi], rtol=1e-6)


@pytest.mark.skipif(not importlib.util.find_spec("openpi_client"), reason="openpi-client is optional")
def test_prepare_image_rotates_and_resizes():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]
    result = pi05._prepare_image(image)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    assert result[-1, -1, 0] > 200
