"""Apple GPU (Metal via MLX) backend for turbovec.

Provides :class:`TurboQuantIndex` running on Apple Silicon GPUs through
MLX. The rotation matrix and Lloyd-Max codebook are sourced from the
Rust crate (``_turbovec.make_rotation_matrix`` /
``_turbovec.codebook``), so ``.tv`` / ``.tvim`` files written by this
backend round-trip bit-exactly with the CPU index.

Phases:
    1. Rotation parity + scaffold (current).
    2. Encode kernel — fused rotate + Lloyd-Max quantize + bit-pack.
    3. Search kernel — fused LUT-build + nibble-scan + top-k.
    4. ``.tv`` / ``.tvim`` load/save + benchmark harness row.
"""
from __future__ import annotations

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError(
        "turbovec.mlx requires the 'mlx' package. "
        "Install with: pip install 'turbovec[mlx]'"
    ) from e

from .._turbovec import codebook as _rust_codebook
from .._turbovec import make_rotation_matrix as _rust_make_rotation_matrix


__all__ = ["TurboQuantIndex"]


class TurboQuantIndex:
    """TurboQuant vector index running on Apple GPU via MLX.

    Mirrors the API of :class:`turbovec.TurboQuantIndex` but executes
    the rotate / quantize / search hot loops as Metal kernels through
    MLX. Currently scaffolding only — ``add`` and ``search`` raise
    ``NotImplementedError`` until the encode and search kernels land
    (phases 2–3).
    """

    def __init__(self, dim: int, bit_width: int) -> None:
        if bit_width not in (2, 4):
            raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")
        self._dim = dim
        self._bit_width = bit_width
        self._n = 0

        rotation_np = _rust_make_rotation_matrix(dim)
        boundaries_np, centroids_np = _rust_codebook(bit_width, dim)
        self._rotation = mx.array(rotation_np)
        self._boundaries = mx.array(boundaries_np)
        self._centroids = mx.array(centroids_np)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bit_width(self) -> int:
        return self._bit_width

    def __len__(self) -> int:
        return self._n

    def _rotate(self, vectors: "mx.array") -> "mx.array":
        """Apply the shared rotation: ``vectors @ R.T``.

        ``vectors`` is ``(n, dim)`` row-major; result is ``(n, dim)``.
        """
        return vectors @ self._rotation.T

    def add(self, vectors) -> None:
        raise NotImplementedError("MLX encode kernel — phase 2")

    def search(self, queries, k: int):
        raise NotImplementedError("MLX search kernel — phase 3")
