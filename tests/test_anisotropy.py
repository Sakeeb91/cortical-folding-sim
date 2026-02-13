"""Tests for anisotropic growth extensions."""

import jax.numpy as jnp

from cortical_folding.synthetic import (
    create_icosphere,
    create_regional_anisotropy,
    create_uniform_anisotropy,
)


def test_anisotropy_test_scaffold():
    verts, _ = create_icosphere(subdivisions=1, radius=1.0)
    assert verts.shape[1] == 3


def test_uniform_anisotropy_shape_and_value():
    field = create_uniform_anisotropy(12, value=0.25)
    assert field.shape == (12,)
    assert jnp.allclose(field, 0.25)


def test_regional_anisotropy_field_has_two_regions():
    verts, faces = create_icosphere(subdivisions=2, radius=1.0)
    field = create_regional_anisotropy(
        verts, faces, high_value=1.0, low_value=0.0, axis=2, threshold=0.0
    )
    assert float(jnp.max(field)) == 1.0
    assert float(jnp.min(field)) == 0.0
