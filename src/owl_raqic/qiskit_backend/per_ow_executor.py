from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np

from owl.science.counter_rng import uniform01
from owl_raqic.math.action_graph import action_graph_hash

from .aer_evidence import parse_aer_gpu_evidence as parse_aer_gpu_evidence
from .aer_runtime import run_aer_job
from .circuit_families import CIRCUIT_FAMILIES, build_circuit_family
from .native_state_preparation import (
    RequiredNativeRuntimeBindingError,
    preflight_native_runtime_binding,
)
from .parameterized_templates import (
    amplitude_bindings,
    build_native_feature_template,
    statevector_action_probabilities,
    supports_runtime_parameter_binding,
)
from .qiskit_policy import QiskitExecutionPolicy, QiskitReadoutPolicy
from .result_decode import ActionBitLayout, counts_to_action_probabilities
from .template_cache import CircuitTemplateCache, template_key
from .workload_planner import estimate_qiskit_workload


@dataclass(frozen=True)
class AuthorityAudit:
    legal_basis: tuple[int, ...]
    illegal_probability: float
    repaired_rest: bool
    passed: bool


@dataclass
class QiskitChunkLedger:
    family: str
    source_start: int
    source_stop: int
    expected_ids: list[int]
    returned_ids: list[int]
    circuit_hashes: list[str]
    metadata: list[dict[str, Any]]
    gpu_verified: bool


@dataclass
class QiskitBatchResult:
    probabilities: np.ndarray
    readouts: np.ndarray
    processed_ow_ids: np.ndarray
    circuit_family: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerOWExecutionResult:
    authoritative: QiskitBatchResult
    families: dict[str, QiskitBatchResult]
    expected_ow_ids: np.ndarray
    workload: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "authoritative_family": self.authoritative.circuit_family,
            "processed_count": int(self.authoritative.processed_ow_ids.size),
            "expected_count": int(self.expected_ow_ids.size),
            "families": {
                name: {
                    "processed_count": int(result.processed_ow_ids.size),
                    "metadata": result.metadata,
                }
                for name, result in self.families.items()
            },
            "workload": self.workload,
            "metadata": self.metadata,
        }


def _row_signature(probability: float, phase: str, mask: Any) -> str:
    payload = np.concatenate(
        [
            np.asarray(probability, dtype=np.float64).view(np.uint8),
            np.asarray(phase, dtype=np.float64).view(np.uint8),
            np.asarray(mask, dtype=np.uint8).view(np.uint8),
        ]
    )
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _result_metadata(result: Any, index: int) -> dict[str, Any]:
    try:
        rows = getattr(result, "results", None) or []
        return dict(getattr(rows[index], "metadata", {}) or {})
    except Exception:
        return {}


def _normalize_row(row: np.ndarray) -> np.ndarray:
    row = np.maximum(np.asarray(row, dtype=np.float64), 0.0)
    total = float(row.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Qiskit probability row has invalid normalization")
    return row / total


def validate_processed_ow_ids(expected: Any, processed: Any) -> np.ndarray:
    """Validate exact, unique, order-preserving per-OW execution accounting."""
    expected_ids = np.asarray(expected, dtype=np.int64).reshape(-1)
    processed_ids = np.asarray(processed, dtype=np.int64).reshape(-1)
    if np.unique(expected_ids).size != expected_ids.size:
        raise ValueError("expected per-OW identities must be unique")
    if np.unique(processed_ids).size != processed_ids.size:
        raise RuntimeError("per-OW Qiskit execution returned duplicate OW ids")
    if processed_ids.size != expected_ids.size:
        missing = np.setdiff1d(expected_ids, processed_ids, assume_unique=False)
        extra = np.setdiff1d(processed_ids, expected_ids, assume_unique=False)
        raise RuntimeError(
            "per-OW Qiskit execution row-count mismatch: "
            f"expected={expected_ids.size}, processed={processed_ids.size}, "
            f"missing={missing.astype(int).tolist()}, extra={extra.astype(int).tolist()}"
        )
    if not np.array_equal(processed_ids, expected_ids):
        if np.array_equal(np.sort(processed_ids), np.sort(expected_ids)):
            raise RuntimeError("per-OW Qiskit execution reordered OW rows")
        missing = np.setdiff1d(expected_ids, processed_ids, assume_unique=False)
        extra = np.setdiff1d(processed_ids, expected_ids, assume_unique=False)
        raise RuntimeError(
            "per-OW Qiskit execution changed OW identities: "
            f"missing={missing.astype(int).tolist()}, extra={extra.astype(int).tolist()}"
        )
    return processed_ids


def _metadata_reports_gpu(metadata: dict[str, Any]) -> bool:
    return bool(parse_aer_gpu_evidence(metadata)["verified"])


def _transpile_for_aer(circuits: Any, simulator: Any) -> Any:
    """Compile family circuits to the installed Aer target.

    Qiskit circuit-library instructions such as ``StatePreparation`` and
    labeled walk unitaries are not guaranteed to be accepted directly by the
    Aer assembler.  Transpiling against the exact simulator target also makes
    method/version incompatibilities fail before an expensive batch launch.
    """
    from qiskit import transpile

    compiled = transpile(circuits, backend=simulator, optimization_level=0)
    return compiled


def _counter_uniform(seed: int, tick: int, ids: np.ndarray, stream_id: int) -> np.ndarray:
    return np.asarray(
        uniform01(seed, tick, np.asarray(ids, dtype=np.uint64), stream_id, 0, xp=np),
        dtype=np.float64,
    )


def _sample_rows(
    probabilities: np.ndarray,
    ow_ids: np.ndarray,
    *,
    seed: int,
    tick: int,
    stream_id: int,
    policy: QiskitReadoutPolicy,
) -> np.ndarray:
    if policy == QiskitReadoutPolicy.ARGMAX:
        return cast(np.ndarray, np.argmax(probabilities, axis=1).astype(np.int32))
    uniforms = _counter_uniform(seed, tick, ow_ids, stream_id)
    cdf = np.cumsum(probabilities, axis=1)
    cdf[:, -1] = 1.0
    return cast(np.ndarray, np.sum(cdf <= uniforms[:, None], axis=1).astype(np.int32))


class PerOWQiskitExecutor:
    """Execute a Qiskit circuit for every eligible OW.

    The implementation is intentionally explicit about cost and row accounting.
    Circuits are submitted in bounded batches. When runtime parameter binding
    is enabled for the static family, a project-owned exact native rotation-tree
    template is compiled to Aer-supported gates before values are bound. No
    RawFeatureVector or ParameterizedInitialize instruction enters Aer.
    """

    def __init__(self, policy: QiskitExecutionPolicy, *, seed: int = 0) -> None:
        self.policy = policy
        self.seed = int(seed)
        self.cache = CircuitTemplateCache() if policy.cache_templates else None
        self.parameter_templates: dict[tuple[str, int, bool, str, str], Any] = {}
        self.runtime_binding_preflight: dict[str, Any] | None = None

    def _simulator(self, family: str, *, runtime_binding: bool = False) -> Any:
        from qiskit_aer import AerSimulator

        spec = CIRCUIT_FAMILIES[family]
        method = self.policy.method
        if family == "density_noise":
            method = "density_matrix"
        if method not in spec.compatible_methods:
            raise ValueError(
                f"circuit family {family!r} is incompatible with Aer method {method!r}"
            )
        options: dict[str, Any] = {
            "method": method,
            "device": str(self.policy.device).upper(),
            "precision": "double",
            "seed_simulator": self.seed,
        }
        if self.policy.target_gpus:
            options["target_gpus"] = list(self.policy.target_gpus)
        if spec.shot_based:
            options["batched_shots_gpu"] = bool(self.policy.batched_shots_gpu)
        if spec.dynamic:
            options["shot_branching_enable"] = bool(self.policy.shot_branching)
        # Runtime binding is safe to enable; it only takes effect for circuits
        # containing parameters and matching parameter_binds.
        options["runtime_parameter_bind_enable"] = bool(runtime_binding)
        if family == "density_noise":
            from qiskit_aer.noise import NoiseModel, depolarizing_error

            model = NoiseModel()
            error = depolarizing_error(0.001, 1)
            model.add_all_qubit_quantum_error(error, ["x", "sx", "rz", "ry", "h"])
            options["noise_model"] = model
        simulator = AerSimulator(**options)
        available = tuple(str(x).upper() for x in simulator.available_devices())
        available_methods = tuple(str(x) for x in simulator.available_methods())
        if method not in available_methods:
            raise RuntimeError(
                f"Aer method {method!r} is unavailable; available methods are {available_methods}"
            )
        requested_device = str(self.policy.device).upper()
        if requested_device not in available:
            raise RuntimeError(
                f"Aer {requested_device} execution requested, available devices are {available}"
            )
        if self.policy.strict_gpu and requested_device != "GPU":
            raise RuntimeError("strict Qiskit execution requires Aer GPU")
        return simulator, options

    def _build_circuit(self, family: str, p: Any, phase: str, mask: Any) -> Any:
        key = template_key(
            mode=family,
            action_count=len(p),
            mask_pattern=tuple(bool(x) for x in mask),
            layout={"signature": _row_signature(p, phase, mask)},
        )
        family_kwargs = {
            "measure": CIRCUIT_FAMILIES[family].shot_based,
            "rounds": 2,
            "n_positions": max(2, 1 << max(1, math.ceil(math.log2(len(p))))),
            "steps": 3,
            "legal_basis": tuple(int(x) for x in np.flatnonzero(mask)),
        }
        if family == "deferred":
            family_kwargs["feedback_phases"] = phase
        if family == "interference":
            action_names = self.policy.action_names
            if not action_names:
                from owl.core.actions import Action

                action_names = tuple(action.name for action in Action)
            family_kwargs.update(
                {
                    "authority_mask": np.asarray(mask, dtype=bool),
                    "mixer_strength": float(self.policy.interference_mixer_strength),
                    "mixer_trotter_steps": int(self.policy.interference_trotter_steps),
                    "action_names": tuple(action_names),
                    "action_graph_hash": action_graph_hash(tuple(action_names)),
                }
            )

        def builder() -> Any:
            return build_circuit_family(family, p, phase, **family_kwargs)

        if self.cache is None:
            return builder()
        return self.cache.get_or_build(key, builder)

    def _get_parameter_template(
        self,
        family: str,
        action_count: int,
        *,
        measure: bool,
        simulator: Any,
    ) -> Any:
        key = (
            str(family),
            int(action_count),
            bool(measure),
            str(self.policy.method),
            str(self.policy.device).upper(),
        )
        if key not in self.parameter_templates:
            self.parameter_templates[key] = build_native_feature_template(
                action_count,
                simulator=simulator,
                family=family,
                measure=measure,
                method=str(self.policy.method),
                device=str(self.policy.device).upper(),
                precision="double",
            )
        return self.parameter_templates[key]

    def preflight_required_runtime_binding(self, action_count: int) -> dict[str, Any] | None:
        if not self.policy.runtime_parameter_binding:
            return None
        if self.policy.runtime_binding_policy != "required_native":
            raise RequiredNativeRuntimeBindingError(
                "flagship runtime binding must use policy 'required_native'"
            )
        if self.runtime_binding_preflight is None:
            self.runtime_binding_preflight = preflight_native_runtime_binding(
                action_count=int(action_count),
                method=str(self.policy.method),
                device=str(self.policy.device).upper(),
                precision="double",
                strict_gpu=bool(self.policy.strict_gpu),
                tolerance=float(self.policy.runtime_binding_preflight_tolerance),
                batch_size=int(self.policy.runtime_binding_preflight_batch_size),
                seed=int(self.seed),
                simulator_options={
                    "batched_shots_gpu": bool(self.policy.batched_shots_gpu),
                },
            )
        return dict(self.runtime_binding_preflight)

    def _execute_parameterized_chunk(
        self,
        family: str,
        probabilities: np.ndarray,
        phases: np.ndarray,
        *,
        shot_based: bool,
    ) -> Any:
        """Execute one structural template with an OW-row parameter batch."""
        simulator, _options = self._simulator(family, runtime_binding=True)
        if self.policy.runtime_binding_preflight_required:
            self.preflight_required_runtime_binding(probabilities.shape[1])
        template = self._get_parameter_template(
            family,
            probabilities.shape[1],
            measure=shot_based,
            simulator=simulator,
        )
        binds, _amplitudes = amplitude_bindings(
            template,
            probabilities,
            phases,
        )
        result = run_aer_job(
            simulator,
            [template.circuit],
            parameter_binds=[binds],
            shots=self.policy.shots if shot_based else None,
        )
        if not result.success:
            raise RuntimeError(getattr(result, "status", "Aer GPU parameter batch failed"))
        rows: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        counts_rows: list[dict[str, int]] = []
        if shot_based:
            width = max(1, int(math.ceil(math.log2(probabilities.shape[1]))))
            layout = ActionBitLayout(tuple(range(width)), little_endian=True)
            for index in range(probabilities.shape[0]):
                counts = dict(result.get_counts(index))
                rows.append(
                    counts_to_action_probabilities(
                        counts,
                        layout,
                        probabilities.shape[1],
                    )
                )
                counts_rows.append(counts)
                metadata.append(_result_metadata(result, index))
            return np.asarray(rows), metadata, counts_rows
        for index in range(probabilities.shape[0]):
            try:
                state = result.get_statevector(index)
            except Exception:
                state = result.data(index)["statevector"]
            rows.append(
                statevector_action_probabilities(
                    state,
                    action_qubits=template.action_qubits,
                    action_count=probabilities.shape[1],
                )
            )
            metadata.append(_result_metadata(result, index))
        return np.asarray(rows), metadata

    def _execute_exact_chunk(
        self,
        family: str,
        probabilities: np.ndarray,
        phases: np.ndarray,
        masks: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if self.policy.runtime_parameter_binding and supports_runtime_parameter_binding(family):
            return cast(
                tuple[np.ndarray, list[dict[str, Any]]],
                self._execute_parameterized_chunk(
                    family,
                    probabilities,
                    phases,
                    shot_based=False,
                ),
            )
        simulator, options = self._simulator(family, runtime_binding=False)
        circuits = []
        builds = []
        for p, phase, mask in zip(probabilities, phases, masks, strict=True):
            built = self._build_circuit(family, p, phase, mask)
            circuit = built.circuit.remove_final_measurements(inplace=False).copy()
            circuit.save_statevector()
            circuits.append(circuit)
            builds.append(built)
        circuits = list(_transpile_for_aer(circuits, simulator))
        result = run_aer_job(simulator, circuits)
        if not result.success:
            raise RuntimeError(getattr(result, "status", "Aer GPU batch failed"))
        rows = []
        metadata = []
        for index, _circuit in enumerate(circuits):
            state = np.asarray(result.get_statevector(index), dtype=np.complex128)
            layout = builds[index].layout
            if layout is None or not layout.action_qubits:
                raise RuntimeError(f"family {family} did not declare action qubits")
            rows.append(
                statevector_action_probabilities(
                    state,
                    action_qubits=tuple(layout.action_qubits),
                    action_count=probabilities.shape[1],
                )
            )
            metadata.append(_result_metadata(result, index))
        return np.asarray(rows), metadata

    def _execute_shot_chunk(
        self,
        family: str,
        probabilities: np.ndarray,
        phases: np.ndarray,
        masks: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, int]]]:
        if self.policy.runtime_parameter_binding and supports_runtime_parameter_binding(family):
            return cast(
                tuple[np.ndarray, list[dict[str, Any]], list[dict[str, int]]],
                self._execute_parameterized_chunk(
                    family,
                    probabilities,
                    phases,
                    shot_based=True,
                ),
            )
        simulator, options = self._simulator(family, runtime_binding=False)
        builds = [
            self._build_circuit(family, p, phase, mask)
            for p, phase, mask in zip(probabilities, phases, masks, strict=True)
        ]
        circuits = [built.circuit for built in builds]
        circuits = list(_transpile_for_aer(circuits, simulator))
        result = run_aer_job(simulator, circuits, shots=self.policy.shots)
        if not result.success:
            raise RuntimeError(getattr(result, "status", "Aer GPU shot batch failed"))
        rows = []
        metadata = []
        counts_rows = []
        for index in range(len(circuits)):
            counts = dict(result.get_counts(index))
            rows.append(
                CIRCUIT_FAMILIES[family].decode_counts(
                    counts,
                    builds[index],
                    probabilities.shape[1],
                    authority=masks[index],
                )
            )
            metadata.append(_result_metadata(result, index))
            counts_rows.append(counts)
        return np.asarray(rows), metadata, counts_rows

    @staticmethod
    def _authority_audits(
        probabilities: np.ndarray,
        authority: np.ndarray,
        *,
        tolerance: float = 1e-8,
        repaired_rest: bool = False,
    ) -> list[AuthorityAudit]:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        authority = np.asarray(authority, dtype=bool)
        if probabilities.shape != authority.shape:
            raise ValueError("probabilities and authority must have matching shapes")
        audits: list[AuthorityAudit] = []
        for row, legal in zip(probabilities, authority, strict=True):
            illegal = float(np.sum(np.where(legal, 0.0, row)))
            audits.append(
                AuthorityAudit(
                    legal_basis=tuple(int(x) for x in np.flatnonzero(legal)),
                    illegal_probability=illegal,
                    repaired_rest=bool(repaired_rest),
                    passed=bool(illegal <= tolerance),
                )
            )
        return audits

    @staticmethod
    def _apply_authority(
        probabilities: np.ndarray,
        authority: np.ndarray,
        *,
        tolerance: float = 1e-8,
    ) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        authority = np.asarray(authority, dtype=bool)
        audits = PerOWQiskitExecutor._authority_audits(
            probabilities, authority, tolerance=tolerance
        )
        failed = [audit for audit in audits if not audit.passed]
        if failed:
            worst = max(audit.illegal_probability for audit in failed)
            raise RuntimeError(f"Qiskit returned probability mass on illegal actions: {worst}")
        masked = np.where(authority, probabilities, 0.0)
        sums = masked.sum(axis=1, keepdims=True)
        if np.any(sums <= 0):
            raise RuntimeError("Qiskit output has no legal probability mass")
        return cast(np.ndarray, masked / sums)

    def execute(
        self,
        probabilities: Any,
        phases: Any,
        authority: Any,
        ow_ids: Any,
        *,
        tick: int,
        tolerance: float = 1e-8,
    ) -> PerOWExecutionResult:
        p = np.asarray(probabilities, dtype=np.float64)
        phase = np.zeros_like(p) if phases is None else np.asarray(phases, dtype=np.float64)
        mask = np.asarray(authority, dtype=bool)
        ids = np.asarray(ow_ids, dtype=np.int64).reshape(-1)
        if p.ndim != 2 or phase.shape != p.shape or mask.shape != p.shape:
            raise ValueError("probability, phase, and authority arrays must have shape [N,A]")
        if ids.size != p.shape[0]:
            raise ValueError("ow_ids length does not match probability rows")
        if np.unique(ids).size != ids.size:
            raise ValueError("per-OW Qiskit execution requires unique OW ids")

        exact_count = sum(CIRCUIT_FAMILIES[name].exact for name in self.policy.circuit_families)
        estimate = estimate_qiskit_workload(
            ow_rows=p.shape[0],
            action_count=p.shape[1],
            family_count=len(self.policy.circuit_families),
            chunk_size=self.policy.chunk_size,
            shots=self.policy.shots,
            exact_family_count=exact_count,
        )
        if (
            estimate.ow_rows * len(self.policy.circuit_families) > 10_000
            and not self.policy.confirm_expensive
        ):
            raise RuntimeError(
                "per-OW Qiskit workload is expensive; set qiskit_confirm_expensive=true"
            )

        family_results: dict[str, QiskitBatchResult] = {}
        for family_index, family in enumerate(self.policy.circuit_families):
            if family not in CIRCUIT_FAMILIES:
                raise ValueError(f"unknown circuit family: {family}")
            prob_chunks = []
            id_chunks = []
            meta_chunks: list[dict[str, Any]] = []
            counts_chunks: list[dict[str, int]] = []
            authority_audits: list[AuthorityAudit] = []
            chunks = [
                (start, min(p.shape[0], start + self.policy.chunk_size))
                for start in range(0, p.shape[0], self.policy.chunk_size)
            ]

            def execute_chunk(bounds: Any, family_name: str = family) -> Any:
                start, stop = bounds
                if CIRCUIT_FAMILIES[family_name].shot_based:
                    q, metadata, counts = self._execute_shot_chunk(
                        family_name, p[start:stop], phase[start:stop], mask[start:stop]
                    )
                else:
                    q, metadata = self._execute_exact_chunk(
                        family_name, p[start:stop], phase[start:stop], mask[start:stop]
                    )
                    counts = []
                gpu_verified = bool(
                    metadata and all(_metadata_reports_gpu(row) for row in metadata)
                )
                if self.policy.strict_gpu and not gpu_verified:
                    raise RuntimeError(
                        f"Aer authoritative {family_name} chunk lacks positive "
                        "GPU execution metadata"
                    )
                audits = self._authority_audits(
                    q,
                    mask[start:stop],
                    tolerance=tolerance,
                    repaired_rest=bool(CIRCUIT_FAMILIES[family_name].noisy),
                )
                q = self._apply_authority(q, mask[start:stop], tolerance=tolerance)
                return start, stop, q, metadata, counts, audits

            # Bound outstanding Aer work. Results are sorted by source row so
            # asynchronous completion cannot reorder OW identities.
            from concurrent.futures import ThreadPoolExecutor

            depth = max(1, int(self.policy.job_queue_depth))
            completed: list[Any] = []
            for window in range(0, len(chunks), depth):
                batch = chunks[window : window + depth]
                if depth == 1:
                    completed.extend(execute_chunk(bounds) for bounds in batch)
                else:
                    with ThreadPoolExecutor(max_workers=depth) as pool:
                        futures = [pool.submit(execute_chunk, bounds) for bounds in batch]
                        completed.extend(future.result() for future in futures)
            for start, stop, q, metadata, counts, audits in sorted(completed):
                prob_chunks.append(q)
                id_chunks.append(ids[start:stop])
                meta_chunks.extend(metadata)
                counts_chunks.extend(counts)
                authority_audits.extend(audits)
            q = np.concatenate(prob_chunks, axis=0) if prob_chunks else np.zeros_like(p)
            processed = np.concatenate(id_chunks) if id_chunks else ids[:0]
            processed = validate_processed_ow_ids(ids, processed)
            readout = _sample_rows(
                q,
                ids,
                seed=self.seed,
                tick=int(tick),
                stream_id=10_000 + family_index,
                policy=self.policy.readout_policy,
            )
            family_results[family] = QiskitBatchResult(
                probabilities=q,
                readouts=readout,
                processed_ow_ids=processed,
                circuit_family=family,
                metadata={
                    "method": self.policy.method,
                    "device": str(self.policy.device).upper(),
                    "shots": self.policy.shots if CIRCUIT_FAMILIES[family].shot_based else None,
                    "chunk_size": self.policy.chunk_size,
                    "job_queue_depth": self.policy.job_queue_depth,
                    "row_metadata": meta_chunks,
                    "cache": None if self.cache is None else self.cache.stats(),
                    "runtime_parameter_binding_requested": self.policy.runtime_parameter_binding,
                    "runtime_parameter_binding_used": bool(
                        self.policy.runtime_parameter_binding
                        and supports_runtime_parameter_binding(family)
                    ),
                    "parameterization_strategy": (
                        "exact_native_rotation_tree"
                        if (
                            self.policy.runtime_parameter_binding
                            and supports_runtime_parameter_binding(family)
                        )
                        else "family_specific_circuit_builder"
                    ),
                    "runtime_parameter_binding_policy": self.policy.runtime_binding_policy,
                    "runtime_parameter_binding_preflight": self.runtime_binding_preflight,
                    "runtime_parameter_binding_fallback_reason": None,
                    "automatic_fallback_allowed": False,
                    "automatic_fallback_used": False,
                    "gpu_execution_verified": bool(
                        meta_chunks and all(_metadata_reports_gpu(row) for row in meta_chunks)
                    ),
                    "counts_rows": counts_chunks if counts_chunks else None,
                    "authority_audit": {
                        "rows": [asdict(item) for item in authority_audits],
                        "max_illegal_probability": (
                            max(
                                (item.illegal_probability for item in authority_audits), default=0.0
                            )
                        ),
                        "all_passed": all(item.passed for item in authority_audits),
                        "projection_policy": (
                            "invalid_or_illegal_to_rest"
                            if CIRCUIT_FAMILIES[family].noisy
                            else "legal_subspace_required"
                        ),
                    },
                    "oracle_metrics": {
                        "max_abs": float(
                            np.max(
                                np.abs(
                                    q
                                    - np.asarray(
                                        [
                                            CIRCUIT_FAMILIES[family].oracle(
                                                row,
                                                steps=3,
                                                n_positions=max(
                                                    2, 1 << max(1, math.ceil(math.log2(len(row))))
                                                ),
                                                depolarizing_probability=0.001,
                                                legal_basis=tuple(
                                                    int(x) for x in np.flatnonzero(row_mask)
                                                ),
                                            )
                                            for row, row_mask in zip(p, mask, strict=True)
                                        ],
                                        dtype=np.float64,
                                    )
                                )
                            )
                        )
                        if q.size
                        else 0.0,
                    },
                },
            )

        authoritative = family_results[self.policy.authoritative_family]
        return PerOWExecutionResult(
            authoritative=authoritative,
            families=family_results,
            expected_ow_ids=ids,
            workload=estimate.to_dict(),
            metadata={
                "strict_gpu": self.policy.strict_gpu,
                "all_ow_accounted": bool(
                    np.array_equal(
                        validate_processed_ow_ids(ids, authoritative.processed_ow_ids), ids
                    )
                ),
                "requested_device": str(self.policy.device).upper(),
                "gpu_execution_verified": bool(
                    family_results
                    and all(
                        bool(item.metadata.get("gpu_execution_verified"))
                        for item in family_results.values()
                    )
                ),
            },
        )
