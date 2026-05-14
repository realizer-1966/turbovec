"""Phase 1 parity tests for the turbovec.mlx Apple/Metal backend.

These cover the Rust -> Python -> MLX bridge only:
  * The Rust-supplied rotation matrix is actually orthogonal.
  * MLX matmul agrees with numpy matmul on the same R and same input
    (to within fp32 reduction-order noise).
  * The Lloyd-Max codebook has the right shape and monotonic boundaries.

They do NOT cover end-to-end encode/search correctness — that arrives
with the Metal kernels in phases 2 and 3.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from turbovec._turbovec import codebook as rust_codebook
from turbovec._turbovec import make_rotation_matrix as rust_make_rotation_matrix
from turbovec.mlx import TurboQuantIndex


@pytest.mark.parametrize("dim", [64, 200, 384, 1536])
def test_rust_rotation_matrix_is_orthogonal(dim):
    R = rust_make_rotation_matrix(dim)
    assert R.shape == (dim, dim)
    assert R.dtype == np.float32

    identity = R @ R.T
    np.testing.assert_allclose(identity, np.eye(dim, dtype=np.float32), atol=1e-4)


@pytest.mark.parametrize("dim", [64, 200, 1536])
def test_mlx_rotation_matches_numpy(dim):
    R = rust_make_rotation_matrix(dim)
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((8, dim)).astype(np.float32)
    x_np /= np.linalg.norm(x_np, axis=1, keepdims=True)

    expected = x_np @ R.T

    index = TurboQuantIndex(dim=dim, bit_width=4)
    x_mx = mx.array(x_np)
    actual = np.asarray(index._rotate(x_mx))

    np.testing.assert_allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 1536])
def test_codebook_shapes_and_monotonic(bit_width, dim):
    boundaries, centroids = rust_codebook(bit_width, dim)
    n_levels = 1 << bit_width

    assert boundaries.shape == (n_levels - 1,)
    assert centroids.shape == (n_levels,)
    assert np.all(np.diff(boundaries) > 0), "boundaries must be strictly increasing"
    assert np.all(np.diff(centroids) > 0), "centroids must be strictly increasing"


def test_index_construction_rejects_bad_bit_width():
    with pytest.raises(ValueError):
        TurboQuantIndex(dim=64, bit_width=3)


def test_index_phase1_stubs_raise():
    index = TurboQuantIndex(dim=64, bit_width=4)
    assert index.dim == 64
    assert index.bit_width == 4
    assert len(index) == 0
    with pytest.raises(NotImplementedError):
        index.add(np.zeros((1, 64), dtype=np.float32))
    with pytest.raises(NotImplementedError):
        index.search(np.zeros((1, 64), dtype=np.float32), k=10)
