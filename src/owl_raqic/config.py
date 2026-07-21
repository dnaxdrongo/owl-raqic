from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .adelic.padic import is_prime


class ActivePlaceConfig(BaseModel):
    include_infinity: bool = True
    primes: tuple[int, ...] = (2, 3, 5)
    prime_weights: dict[int, float] = Field(default_factory=lambda: {2: 0.25, 3: 0.15, 5: 0.10})
    modulus_power: int = 8

    @field_validator("primes")
    @classmethod
    def primes_are_prime(cls, v: Any) -> Any:
        if len(set(v)) != len(v):
            raise ValueError("active primes must be unique")
        for p in v:
            if not is_prime(p):
                raise ValueError(f"{p} is not prime")
        return v

    @model_validator(mode="after")
    def weights_finite(self) -> Any:
        for p in self.primes:
            if p not in self.prime_weights:
                self.prime_weights[p] = 1.0 / max(1, len(self.primes))
        for w in self.prime_weights.values():
            if not (float("-inf") < float(w) < float("inf")):
                raise ValueError("prime weights must be finite")
        if self.modulus_power < 1:
            raise ValueError("modulus_power must be positive")
        return self


class RAQICRegisterConfig(BaseModel):
    n_scale: int = 3
    n_places: int = 4
    n_features: int = 8
    n_actions: int = 10
    n_readouts: int = 2
    use_action_mask: bool = True

    @field_validator("n_scale", "n_places", "n_features", "n_actions", "n_readouts")
    @classmethod
    def positive(cls, v: Any) -> Any:
        if v <= 0:
            raise ValueError("register sizes/counts must be positive")
        return v


class RAQICAlgorithmConfig(BaseModel):
    mode: Literal["static", "dynamic", "deferred", "hybrid", "walk", "cpu_audit"] = "cpu_audit"
    rounds: int = 1
    shots: int = 4096
    seed: int = 1234
    beta_intention: float = 1.0
    epsilon_adelic: float = 1.0
    backend_policy: Literal[
        "auto", "statevector", "density_matrix", "sampler", "mps", "cpu_audit"
    ] = "cpu_audit"
    action_temperature: float = 1.0
    phase_mode: Literal["scalar_reference", "canonical_device"] = "scalar_reference"
    memory_limit_mb: float = 512.0
    allow_large_simulation: bool = False
    active_places: ActivePlaceConfig = Field(default_factory=ActivePlaceConfig)
    registers: RAQICRegisterConfig = Field(default_factory=RAQICRegisterConfig)

    @field_validator("rounds")
    @classmethod
    def rounds_nonnegative(cls, v: Any) -> Any:
        if v < 0:
            raise ValueError("rounds must be nonnegative")
        return v

    @field_validator("shots")
    @classmethod
    def shots_positive(cls, v: Any) -> Any:
        if v <= 0:
            raise ValueError("shots must be positive")
        return v

    @field_validator("action_temperature")
    @classmethod
    def temperature_positive(cls, v: Any) -> Any:
        if v <= 0:
            raise ValueError("action_temperature must be positive")
        return v


class BackendProfile(BaseModel):
    name: str
    simulator: str = "AerSimulator"
    method: Literal[
        "statevector",
        "density_matrix",
        "sampler",
        "matrix_product_state",
        "tensor_network",
        "cpu_audit",
    ] = "cpu_audit"
    device: Literal["CPU", "GPU", "NONE"] = "NONE"
    optional: bool = False
    qiskit_required: bool = False
    gpu_required: bool = False
