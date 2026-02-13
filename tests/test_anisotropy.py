"""Tests for anisotropic growth extensions."""

import jax.numpy as jnp

from cortical_folding.synthetic import create_icosphere


def test_anisotropy_test_scaffold():
    verts, _ = create_icosphere(subdivisions=1, radius=1.0)
    assert verts.shape[1] == 3
