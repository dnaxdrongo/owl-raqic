from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldCoverage:
    category: str
    behavioral_tests: tuple[str, ...]
    runtime_contract: str


def coverage_for_field(name: str) -> FieldCoverage:
    """Return the maintained behavioral-test contract for a RAQIC field.

    This ledger complements the AST runtime-usage audit. The named test files
    exercise the subsystem contract, while the all-config test exercises schema
    and migration behavior for every shipped configuration.
    """
    if name.startswith("full_gpu_"):
        return FieldCoverage(
            "full_gpu",
            (
                "tests/test_v09_production_closeout.py",
                "tests/test_gpu_full_loop_optional.py",
            ),
            "persistent/graph/distributed/visual runtime",
        )
    if name.startswith("qiskit_") or name in {
        "use_qiskit_for_all",
        "qiskit_subset_fraction",
        "cache_templates",
        "batch_by_feature_signature",
        "shots",
    }:
        return FieldCoverage(
            "qiskit",
            (
                "tests/test_v09_production_closeout.py",
                "tests/test_gpu_v08_qiskit_strict.py",
                "tests/test_standalone_raqic_qiskit_backend_optional.py",
            ),
            "Qiskit execution or validation policy",
        )
    if name.startswith("gpu_") or name in {
        "dense_signature_grouping",
        "max_cells_per_tick",
    }:
        return FieldCoverage(
            "dense_gpu",
            (
                "tests/test_v09_production_closeout.py",
                "tests/test_raqic_gpu_integration.py",
                "tests/test_raqic_gpu_optional.py",
            ),
            "dense RAQIC GPU execution",
        )
    return FieldCoverage(
        "core_raqic",
        (
            "tests/test_raqic_integration.py",
            "tests/test_raqic_dense_equivalence.py",
            "tests/test_standalone_raqic_driver.py",
        ),
        "CPU/scalar/dense RAQIC recovery behavior",
    )
