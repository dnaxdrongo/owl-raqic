#!/usr/bin/env python3
"""Independently certify CADC-MORE 2 artifacts while keeping confirmatory evaluation locked."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json, sha256_file  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.contracts import load_and_validate_certificate  # noqa: E402
from owl.cadc.features import (  # noqa: E402
    FeatureRegistry,
    validate_feature_perspective,
)
from owl.cadc.schema import (  # noqa: E402
    ACTION_FAMILY_REGISTRY,
    PHASE4_CERTIFICATE_VERSION,
    FeaturePerspective,
    FeatureStage,
)
from owl.experiments.controller import _release_hash  # noqa: E402


def _load(path: str | Path, name: str) -> dict[str, Any]:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"{name} evidence missing: {value}")
    payload = json.loads(value.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} evidence must be a JSON object")
    return payload


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def certify(args: argparse.Namespace) -> dict[str, Any]:
    config = load_phase4_config(args.config)
    provenance = load_and_validate_certificate(config.phase3_input.phase3_certificate)
    corpus = _load(args.corpus_certificate, "corpus")
    dataset = _load(args.dataset_receipt, "dataset")
    repeat_pilot = _load(args.repeat_pilot, "repeat pilot")
    training = _load(args.training_receipt, "training")
    calibration = _load(args.calibration_receipt, "calibration")
    scored = _load(args.score_receipt, "scored artifacts")
    evaluation = _load(args.evaluation, "evaluation")
    controls = _load(args.negative_controls, "negative controls")
    math = _load(args.math_verification, "math verification")
    casebook = _load(args.casebook_manifest, "casebook")
    environment = _load(args.environment_manifest, "environment")
    gpu_stack_smoke = _load(args.gpu_stack_smoke, "GPU stack smoke")
    performance = _load(args.performance, "performance")
    hotpath = _load(args.hotpath_audit, "hot-path audit")
    synthetic = _load(args.synthetic_scenarios, "synthetic scenarios")
    commands = _load(args.command_status, "command status")
    registry = FeatureRegistry()
    for definition in registry.definitions:
        validate_feature_perspective(definition)
    primary_features = registry.for_perspective(FeaturePerspective.AGENT_PRIMARY)
    mechanism_features = registry.for_perspective(
        FeaturePerspective.MECHANISM_MEDIATION
    )
    mechanism_exclusion = bool(primary_features) and bool(mechanism_features) and all(
        definition.stage is FeatureStage.PRE_CHOICE
        and definition.perspective is FeaturePerspective.AGENT_PRIMARY
        for definition in primary_features
    ) and not {
        definition.name for definition in primary_features
    }.intersection(definition.name for definition in mechanism_features)
    failures = []
    checks = {
        "phase3_source": provenance.phase3_source_sha256
        == "d17ef58692c7663eb0cc87ab4cdf7e74ca9b529091fcab4f15b6fe28e2a607a3",
        "corpus": corpus.get("passed") is True,
        "dataset": dataset.get("passed") is True,
        "repeat_pilot": repeat_pilot.get("passed") is True,
        "training": training.get("passed") is True,
        "calibration": calibration.get("passed") is True,
        "scored_artifacts": scored.get("passed") is True,
        "evaluation": evaluation.get("passed") is True,
        "negative_controls": controls.get("passed") is True,
        "math": math.get("passed") is True,
        "casebook": casebook.get("passed") is True and casebook.get("blinded") is True,
        "environment": environment.get("passed") is True,
        "gpu_stack_smoke": gpu_stack_smoke.get("passed") is True
        and (
            (
                config.runtime.target.value == "cpu"
                and gpu_stack_smoke.get("skipped") is True
            )
            or (
                config.runtime.target.value != "cpu"
                and gpu_stack_smoke.get("skipped") is False
                and gpu_stack_smoke.get("dlpack_zero_copy") is True
                and gpu_stack_smoke.get("torch_bf16") is True
                and gpu_stack_smoke.get("xgboost_cuda") is True
                and gpu_stack_smoke.get("cuml_cuda") is True
                and gpu_stack_smoke.get("support_geometry_cuda_float64") is True
            )
        ),
        "performance": performance.get("passed") is True,
        "hotpath": hotpath.get("passed") is True,
        "synthetic_scenarios": synthetic.get("passed") is True,
        "synthetic_case_count": synthetic.get("case_count") == 15
        and len(synthetic.get("checks", [])) == 15,
        "mechanism_exclusion": mechanism_exclusion,
        "commands": bool(commands) and all(int(value) == 0 for value in commands.values()),
        "finite_evaluation": _finite_tree(evaluation),
        "phase5_lock_config": config.certification.phase5_unlock_requested is False,
        "corpus_contract": corpus.get("corpus_contract_sha256")
        == config.corpus_digest(),
        "dataset_contract": dataset.get("corpus_contract_sha256")
        == config.corpus_digest(),
        "dataset_model_spec": dataset.get("model_spec_sha256")
        == config.model_spec_digest(),
        "training_model_spec": training.get("model_spec_sha256")
        == config.model_spec_digest(),
        "scored_model_spec": scored.get("model_spec_sha256")
        == config.model_spec_digest(),
        "environment_model_spec": environment.get("model_spec_sha256")
        == config.model_spec_digest(),
        "precision_contract": training.get("precision") == config.runtime.precision
        and performance.get("precision") == config.runtime.precision
        and environment.get("precision") == config.runtime.precision,
        "phase5_locked_everywhere": all(
            evidence.get("phase5_locked") is True
            for evidence in (
                corpus,
                dataset,
                repeat_pilot,
                training,
                calibration,
                scored,
                evaluation,
                controls,
                casebook,
                environment,
                gpu_stack_smoke,
                performance,
                synthetic,
            )
        ),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    if config.runtime.precision == "fp8":
        failures.append("b200_fp8_parity_path_not_certified")
    target = config.runtime.target.value.upper()
    expected_device = {
        "CPU": None,
        "H100": "H100",
        "H200": "H200",
        "B200": "B200",
    }[target]
    if expected_device is not None and expected_device not in str(
        performance.get("device", "")
    ).upper():
        failures.append("target_device_identity")
    if performance.get("target", "").upper() != target:
        failures.append("performance_target")
    if environment.get("target", "").upper() != target:
        failures.append("environment_target")
    expected_folds = config.splits.outer_folds
    if len(evaluation.get("folds", [])) != expected_folds:
        failures.append("outer_fold_evidence_count")
    suite_models = [
        value
        for value in training.get("models", [])
        if value.get("role") == "cadc_more2_suite"
    ]
    if len(suite_models) != expected_folds * config.models.ensemble_members:
        failures.append("ensemble_member_evidence_count")
    neural_baselines = [
        value
        for value in training.get("models", [])
        if value.get("role") == "viability_baseline"
    ]
    if len(neural_baselines) != expected_folds:
        failures.append("neural_baseline_evidence_count")
    if config.models.xgboost_enabled:
        tree_baselines = [
            value
            for value in training.get("models", [])
            if value.get("role") == "viability_baseline_xgboost"
            and value.get("context_only") is True
        ]
        if len(tree_baselines) != expected_folds:
            failures.append("xgboost_baseline_evidence_count")
        survival_baselines = [
            value
            for value in training.get("models", [])
            if value.get("role")
            == "survival_baseline_xgboost_action_agnostic"
        ]
        if len(survival_baselines) != expected_folds:
            failures.append("xgboost_survival_baseline_evidence_count")
        for role in (
            "candidate_ranker_xgboost_agent",
            "candidate_ranker_xgboost_oracle",
        ):
            rankers = [
                value for value in training.get("models", []) if value.get("role") == role
            ]
            if len(rankers) != expected_folds:
                failures.append(f"{role}_evidence_count")
    phase4_source_sha256 = _release_hash(ROOT)
    dataset_validation = dataset.get("validation", {})
    checks.update(
        {
            "phase4_source_chain": dataset.get("phase4_source_sha256")
            == phase4_source_sha256
            and training.get("phase4_source_sha256") == phase4_source_sha256
            and environment.get("phase4_source_sha256") == phase4_source_sha256,
            "dataset_validation": isinstance(dataset_validation, dict)
            and dataset_validation.get("passed") is True,
            "repeat_policy": repeat_pilot.get("selected_repeat_policy")
            == config.corpus.repeat_policy
            and repeat_pilot.get("corpus_contract_sha256")
            == config.corpus_digest(),
            "sense_flee_pursue_support": isinstance(dataset_validation, dict)
            and dataset_validation.get("sense_flee_pursue_supported") is True,
            "confirmatory_seed_seal": corpus.get("sealed_phase5_seeds")
            == list(config.corpus.reserved_phase5_seeds)
            and corpus.get("sealed_phase6_seeds")
            == list(config.corpus.reserved_phase6_seeds),
            "negative_control_classification": controls.get("classification")
            == "NEGATIVE_CONTROLS_COLLAPSED",
            "epistemic_evidence_boundary": training.get("component_status", {}).get(
                "epistemic_later_action_change"
            )
            == "unsupported_evidence"
            and "epistemic_later_action_change"
            in training.get("unsupported_evidence", {}),
        }
    )
    failures.extend(
        name for name, passed in checks.items() if not passed and name not in failures
    )
    required_families = {
        value.primary_family.value for value in ACTION_FAMILY_REGISTRY
    }
    if set(casebook.get("reviewer_challenge_family_coverage", [])) != required_families:
        failures.append("reviewer_challenge_family_coverage")
    challenge_path = Path(args.casebook_manifest).resolve().parent / (
        "reviewer_challenge_set.jsonl"
    )
    if (
        not challenge_path.is_file()
        or sha256_file(challenge_path) != casebook.get("reviewer_challenge_sha256")
        or int(casebook.get("reviewer_challenge_count", 0)) <= 0
    ):
        failures.append("reviewer_challenge_checksum")
    for fold in evaluation.get("folds", []):
        subgroups = fold.get("subgroups", {})
        if set(subgroups.get("action_family", {})) != required_families:
            failures.append("action_family_subgroup_coverage")
            break
    reload_tolerance = 1e-6 if config.runtime.precision == "fp32" else 5e-3
    if any(
        float(model.get("reload_max_abs_error", math.inf)) > reload_tolerance
        for model in training.get("models", [])
    ):
        failures.append("model_reload_identity")
    artifact_checksums = True
    for model in training.get("models", []):
        path = Path(str(model.get("path", "")))
        if (
            not path.is_file()
            or not model.get("sha256")
            or sha256_file(path) != model.get("sha256")
        ):
            artifact_checksums = False
            break
    if artifact_checksums:
        histories = training.get("training_histories", [])
        if len(histories) != expected_folds:
            artifact_checksums = False
        for history in histories:
            path = Path(str(history.get("path", "")))
            if (
                not path.is_file()
                or sha256_file(path) != history.get("sha256")
                or path.stat().st_size != int(history.get("bytes", -1))
                or int(history.get("rows", 0)) <= 0
            ):
                artifact_checksums = False
                break
    dataset_root = Path(args.dataset_receipt).resolve().parent / "canonical_data"
    if artifact_checksums:
        for part in dataset.get("parts", []):
            path = dataset_root / str(part.get("name", "")) / "part-000000.parquet"
            if (
                not path.is_file()
                or sha256_file(path) != part.get("sha256")
                or path.stat().st_size != int(part.get("bytes", -1))
            ):
                artifact_checksums = False
                break
    score_root = Path(args.score_receipt).resolve().parent
    for name, field in (
        ("candidate_scores_compact.parquet", "candidate_sha256"),
        ("decision_scores_compact.parquet", "decision_sha256"),
    ):
        path = score_root / name
        if not path.is_file() or sha256_file(path) != scored.get(field):
            artifact_checksums = False
    for fold in calibration.get("folds", []):
        path = Path(str(fold.get("support_index", "")))
        if not path.is_file() or sha256_file(path) != fold.get("support_index_sha256"):
            artifact_checksums = False
    if (
        not challenge_path.is_file()
        or sha256_file(challenge_path) != casebook.get("reviewer_challenge_sha256")
    ):
        artifact_checksums = False
    checks["artifact_checksums"] = artifact_checksums
    if not artifact_checksums:
        failures.append("artifact_checksums")
    fold_roots = {
        Path(str(model["path"])).parent
        for model in training.get("models", [])
        if model.get("role") == "cadc_more2_suite" and model.get("path")
    }
    if len(fold_roots) != config.splits.outer_folds or any(
        not (root / "model_card.md").is_file()
        or not (root / "ensemble_manifest.json").is_file()
        for root in fold_roots
    ):
        failures.append("model_artifact_contract")
    required_packages = environment.get("required_versions", {})
    for package in (
        "numpy",
        "pyarrow",
        "polars",
        "scikit-learn",
        "torch",
        "xgboost",
    ):
        if not required_packages.get(package):
            failures.append(f"environment_package:{package}")
    if config.runtime.target.value != "cpu":
        if any(
            fold.get("support", {}).get("neighbor_backend") != "cuml_cuda"
            or fold.get("support", {}).get("geometry_backend")
            != "cupy_cuda_float64"
            for fold in calibration.get("folds", [])
        ):
            failures.append("gpu_support_index_backend")
        if not all(
            (
                performance.get("dlpack_zero_copy", {}).get("context") is True,
                performance.get("dlpack_zero_copy", {}).get("candidates") is True,
                performance.get("dlpack_zero_copy", {}).get("directions") is True,
                performance.get("within_device_memory_bound") is True,
                float(performance.get("fp32_cpu_gpu_max_abs_error", math.inf)) <= 2e-4,
                float(performance.get("batch_single_max_abs_error", math.inf)) <= 1e-5,
                float(performance.get("deterministic_repeat_max_abs_error", math.inf))
                <= 1e-7,
            )
        ):
            failures.append("target_gpu_parity_or_memory")
        if config.runtime.compile and (
            performance.get("compiled_path_exercised") is not True
            or float(performance.get("eager_compiled_max_abs_error", math.inf))
            > reload_tolerance
        ):
            failures.append("compiled_eager_parity")
    elif config.certification.require_target_gpu:
        failures.append("target_gpu_required_but_cpu_configured")
    classification = (
        f"{target}_PHASE4_DEVELOPMENT_CANDIDATE" if not failures else "FAILED_CLOSED"
    )
    evidence = {
        "phase3_certificate": sha256_file(config.phase3_input.phase3_certificate),
        "corpus_certificate": sha256_file(args.corpus_certificate),
        "dataset_receipt": sha256_file(args.dataset_receipt),
        "repeat_pilot": sha256_file(args.repeat_pilot),
        "training_receipt": sha256_file(args.training_receipt),
        "calibration_receipt": sha256_file(args.calibration_receipt),
        "score_receipt": sha256_file(args.score_receipt),
        "evaluation": sha256_file(args.evaluation),
        "negative_controls": sha256_file(args.negative_controls),
        "math_verification": sha256_file(args.math_verification),
        "casebook_manifest": sha256_file(args.casebook_manifest),
        "environment_manifest": sha256_file(args.environment_manifest),
        "gpu_stack_smoke": sha256_file(args.gpu_stack_smoke),
        "performance": sha256_file(args.performance),
        "hotpath_audit": sha256_file(args.hotpath_audit),
        "synthetic_scenarios": sha256_file(args.synthetic_scenarios),
        "command_status": sha256_file(args.command_status),
    }
    gate_map = {
        "source_identity": checks.get("phase3_source", False),
        "phase3_certificate": provenance.phase3_source_sha256
        == "d17ef58692c7663eb0cc87ab4cdf7e74ca9b529091fcab4f15b6fe28e2a607a3",
        "corpus_certificate": checks.get("corpus", False),
        "schema_identity": checks.get("dataset_model_spec", False),
        "catalog_integrity": checks.get("dataset_validation", False),
        "split_integrity": checks.get("confirmatory_seed_seal", False),
        "agent_oracle_separation": checks.get("epistemic_evidence_boundary", False),
        "mechanism_exclusion": checks.get("mechanism_exclusion", False),
        "training_completion": checks.get("training", False),
        "model_reload": "model_reload_identity" not in failures,
        "crossfit_predictions": checks.get("scored_artifacts", False),
        "forecast_metrics": checks.get("evaluation", False),
        "rank_metrics": checks.get("evaluation", False),
        "survival_metrics": checks.get("evaluation", False),
        "epistemic_metrics": checks.get("epistemic_evidence_boundary", False),
        "calibration": checks.get("calibration", False),
        "support_abstention": checks.get("calibration", False),
        "negative_controls": checks.get("negative_controls", False),
        "family_coverage": "action_family_subgroup_coverage" not in failures,
        "synthetic_scenarios": checks.get("synthetic_scenarios", False),
        "target_gpu_evidence": checks.get("performance", False)
        and checks.get("gpu_stack_smoke", False)
        and (expected_device is None or "target_gpu_parity_or_memory" not in failures),
        "performance_bounds": checks.get("performance", False),
        "quality_toolchain": checks.get("commands", False),
        "artifact_checksums": artifact_checksums,
        "confirmatory_seed_seal": checks.get("confirmatory_seed_seal", False),
    }
    return {
        "schema_version": PHASE4_CERTIFICATE_VERSION,
        "passed": not failures,
        "classification": classification,
        "failures": [
            {"gate": str(value), "classification": "failed_closed"}
            for value in dict.fromkeys(failures)
        ],
        "checks": checks,
        "gates": [
            {
                "gate": name,
                "passed": bool(passed),
                "classification": "passed" if passed else "failed_closed",
            }
            for name, passed in gate_map.items()
        ],
        "phase3_source_sha256": provenance.phase3_source_sha256,
        "phase4_source_sha256": phase4_source_sha256,
        "phase4_config_sha256": config.canonical_digest(),
        "corpus_contract_sha256": config.corpus_digest(),
        "model_spec_sha256": config.model_spec_digest(),
        "dataset_id": dataset.get("dataset_id"),
        "target": config.runtime.target.value,
        "precision": config.runtime.precision,
        "device": performance.get("device", "cpu"),
        "evaluation_aggregate": evaluation.get("aggregate", {}),
        "negative_control_classification": controls.get("classification"),
        "evidence_sha256": evidence,
        "freeze_decision": "NOT_TAKEN",
        "phase5_unlocked": False,
        "phase5_lock_reason": "separate Phase 5 freeze decision required",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus-certificate", required=True)
    parser.add_argument("--dataset-receipt", required=True)
    parser.add_argument("--repeat-pilot", required=True)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--calibration-receipt", required=True)
    parser.add_argument("--score-receipt", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--negative-controls", required=True)
    parser.add_argument("--math-verification", required=True)
    parser.add_argument("--casebook-manifest", required=True)
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--gpu-stack-smoke", required=True)
    parser.add_argument("--performance", required=True)
    parser.add_argument("--hotpath-audit", required=True)
    parser.add_argument("--synthetic-scenarios", required=True)
    parser.add_argument("--command-status", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        certificate = certify(args)
    except Exception as exc:
        gate_names = (
            "source_identity",
            "phase3_certificate",
            "corpus_certificate",
            "schema_identity",
            "catalog_integrity",
            "split_integrity",
            "agent_oracle_separation",
            "mechanism_exclusion",
            "training_completion",
            "model_reload",
            "crossfit_predictions",
            "forecast_metrics",
            "rank_metrics",
            "survival_metrics",
            "epistemic_metrics",
            "calibration",
            "support_abstention",
            "negative_controls",
            "family_coverage",
            "synthetic_scenarios",
            "target_gpu_evidence",
            "performance_bounds",
            "quality_toolchain",
            "artifact_checksums",
            "confirmatory_seed_seal",
        )
        certificate = {
            "schema_version": PHASE4_CERTIFICATE_VERSION,
            "passed": False,
            "classification": "FAILED_CLOSED",
            "failures": [
                {
                    "gate": "certifier_exception",
                    "classification": "failed_closed",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                }
            ],
            "checks": dict.fromkeys(gate_names, False),
            "gates": [
                {
                    "gate": name,
                    "passed": False,
                    "classification": "not_evaluated_after_exception",
                }
                for name in gate_names
            ],
            "traceback": traceback.format_exc().splitlines(),
            "phase5_unlocked": False,
            "freeze_decision": "NOT_TAKEN",
        }
    atomic_json(args.output, certificate)
    return 0 if certificate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
