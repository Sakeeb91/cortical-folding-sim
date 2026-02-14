"""Tests for comparison animation helper."""

from __future__ import annotations

import numpy as np
import pytest

from cortical_folding.viz import save_comparison_animation


def test_save_comparison_animation_rejects_unsupported_suffix(tmp_path):
    baseline = np.zeros((3, 3, 3), dtype=np.float32)
    improved = np.zeros((3, 3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(ValueError, match="comparison outputs"):
        save_comparison_animation(
            baseline,
            improved,
            faces,
            output_paths=[str(tmp_path / "bad.ext")],
        )
