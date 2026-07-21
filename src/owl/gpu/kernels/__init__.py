"""Provide optional CuPy raw and elementwise kernels for GPU optimization.

Each kernel module exposes a vectorized NumPy/CuPy fallback and only compiles raw
CUDA code when CuPy is present and a CUDA backend is active.
"""
