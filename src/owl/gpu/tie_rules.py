from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ThresholdTieRules:
    """Centralized boundary comparisons for GPU/CPU parity.

    These rules make threshold-sensitive ecological decisions explicit instead
    of leaving them as scattered ``<``/``<=`` conventions.
    """

    death_inclusive: bool = True
    viability_inclusive: bool = True
    cdf_strict_greater: bool = True

    def death_mask(self, health: Any, threshold: float, xp: Any) -> Any:
        return health <= threshold if self.death_inclusive else health < threshold

    def viable_mask(self, value: Any, minimum: float, xp: Any) -> Any:
        return value >= minimum if self.viability_inclusive else value > minimum

    def sample_from_cdf(self, cdf: Any, uniforms: Any, xp: Any) -> Any:
        """Return first index where CDF crosses ``u`` using configured tie rule."""
        u = uniforms[..., None]
        crossed = cdf > u if self.cdf_strict_greater else cdf >= u
        return xp.argmax(crossed, axis=-1).astype(xp.int32)


DEFAULT_TIE_RULES = ThresholdTieRules()


def quantize_priority(probability: float, xp: Any, scale: int = 1_000_000) -> Any:
    """Quantize floating priorities before sorting to make ties reproducible."""
    return xp.rint(probability * scale).astype(xp.int64)


def deterministic_mix_u64(x: int) -> int:
    """SplitMix64-style deterministic mixer used by CPU/GPU audit RNG specs."""
    x = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF


def deterministic_uniform01(seed: int, tick: int, cell_id: int, stream_id: int = 0) -> float:
    """Deterministic CPU uniform in [0,1) from integer identifiers."""
    key = (
        (int(seed) & 0xFFFFFFFF)
        ^ ((int(tick) & 0xFFFFFFFF) << 16)
        ^ ((int(stream_id) & 0xFFFF) << 48)
    )
    mixed = deterministic_mix_u64(key ^ int(cell_id))
    # Use 53 bits for a double-precision uniform.
    return ((mixed >> 11) & ((1 << 53) - 1)) / float(1 << 53)
