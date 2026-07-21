"""Construct the canonical provenance-bound CADC-MORE 2 dataset."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from owl.cadc.artifacts import atomic_json, sha256_file
from owl.cadc.catalog import Phase4EvidenceCatalog
from owl.cadc.features import FeatureRegistry
from owl.cadc.gpu_io import GPUParquetReader
from owl.cadc.outcomes import OutcomeRegistry
from owl.cadc.schema import PHASE4_DATASET_SCHEMA_VERSION, stable_id
from owl.cadc.splits import SplitRegistry, validate_no_leakage
from owl.record.cadc_schema import CADCEventCode

DECISION_KEY = ("run_id", "condition", "seed", "tick", "decision_sequence", "ow_id")
SOURCE_JOIN_KEY = ("tick", "decision_sequence", "ow_id")


@dataclass(frozen=True)
class GroupedBatchIndex:
    """Contiguous grouped-row index for candidate/listwise and branch batches."""

    group_ids: npt.NDArray[Any]
    row_order: npt.NDArray[Any]
    offsets: npt.NDArray[Any]

    @classmethod
    def build(cls, group_ids: Any) -> GroupedBatchIndex:
        """Build a stable contiguous row index for repeated group identifiers."""
        values = np.asarray(group_ids).astype(str)
        order = np.argsort(values, kind="stable")
        ordered = values[order]
        starts = (
            np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
            if ordered.size
            else np.asarray([0], dtype=np.int64)
        )
        offsets = np.r_[starts, ordered.size].astype(np.int64)
        unique = ordered[starts] if ordered.size else np.asarray([], dtype=str)
        return cls(unique, order.astype(np.int64), offsets)

    def validate_exact_size(self, expected: int) -> None:
        """Require every indexed group to contain the declared row count."""
        sizes = np.diff(self.offsets)
        if sizes.size and not np.all(sizes == expected):
            bad = self.group_ids[sizes != expected]
            raise ValueError(f"groups do not contain exactly {expected} rows: {bad[:5]}")


@dataclass(frozen=True)
class CanonicalPartitionReceipt:
    """Checksum, cardinality, and derivation receipt for one canonical table."""
    name: str
    path: str
    rows: int
    bytes: int
    sha256: str
    source_columns: tuple[str, ...]
    derivation: str


def _normalized_key(columns: Mapping[str, Any], fields: Sequence[str]) -> npt.NDArray[Any]:
    if not fields:
        raise ValueError("key needs at least one field")
    missing = [field for field in fields if field not in columns]
    if missing:
        raise KeyError(f"key columns missing: {missing}")
    rows = len(columns[fields[0]])
    pieces = []
    for field in fields:
        values = np.asarray(columns[field]).astype(str)
        if values.size != rows:
            raise ValueError("key columns have unequal row counts")
        pieces.append(np.char.add(f"{field}=", values))
    result = pieces[0]
    for piece in pieces[1:]:
        result = np.char.add(np.char.add(result, "\x1f"), piece)
    return result


def join_source_decisions(
    factual_decisions: Mapping[str, Any], source_decisions: Mapping[str, Any]
) -> dict[str, npt.NDArray[Any]]:
    """Join each counterfactual source decision to exactly one factual decision."""
    factual_key = _normalized_key(factual_decisions, SOURCE_JOIN_KEY)
    source_key = _normalized_key(source_decisions, SOURCE_JOIN_KEY)
    order = np.argsort(factual_key, kind="stable")
    ordered = factual_key[order]
    if ordered.size and np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("factual source-join key is not unique inside its run")
    positions = np.searchsorted(ordered, source_key)
    valid = positions < ordered.size
    matched = np.zeros(source_key.size, dtype=bool)
    matched[valid] = ordered[positions[valid]] == source_key[valid]
    if not matched.all():
        raise ValueError(f"source decisions without exact factual join: {int((~matched).sum())}")
    factual_rows = order[positions]
    if "factual_selected_action" in source_decisions and "selected_action" in factual_decisions:
        source_action = np.asarray(source_decisions["factual_selected_action"])
        factual_action = np.asarray(factual_decisions["selected_action"])[factual_rows]
        if not np.array_equal(source_action, factual_action):
            raise ValueError("source/factual selected actions do not match exactly")
    output = {
        name: np.asarray(value)[factual_rows] for name, value in factual_decisions.items()
    }
    output.update({name: np.asarray(value) for name, value in source_decisions.items()})
    return output


class FoldTransformer:
    """Train-fold-only robust centering, scaling, clipping, and missing policy."""

    def __init__(self, *, clip_quantile: float = 0.005, epsilon: float = 1e-6) -> None:
        if not 0.0 <= clip_quantile < 0.5:
            raise ValueError("clip_quantile must be in [0, 0.5)")
        self.clip_quantile = float(clip_quantile)
        self.epsilon = float(epsilon)
        self.parameters: dict[str, dict[str, float]] = {}

    def fit(self, columns: Mapping[str, Any], train_mask: Any) -> FoldTransformer:
        """Fit robust transformations on training rows only."""
        mask = np.asarray(train_mask, dtype=bool)
        if not mask.any():
            raise ValueError("transformer train mask is empty")
        fitted: dict[str, dict[str, float]] = {}
        for name, value in sorted(columns.items()):
            array = np.asarray(value)
            if array.ndim != 1 or array.dtype.kind not in "fiu":
                continue
            train = array[mask].astype(np.float64)
            finite = train[np.isfinite(train)]
            if not finite.size:
                raise ValueError(f"training feature has no finite values: {name}")
            median = float(np.median(finite))
            q25, q75 = np.quantile(finite, (0.25, 0.75))
            scale = max(float(q75 - q25), self.epsilon)
            lower, upper = np.quantile(
                finite, (self.clip_quantile, 1.0 - self.clip_quantile)
            )
            fitted[name] = {
                "median": median,
                "scale": scale,
                "lower": float(lower),
                "upper": float(upper),
                "missing": median,
            }
        self.parameters = fitted
        return self

    def transform(self, columns: Mapping[str, Any]) -> dict[str, npt.NDArray[Any]]:
        """Apply the frozen robust transformation to compatible columns."""
        if not self.parameters:
            raise RuntimeError("fold transformer has not been fit")
        output: dict[str, npt.NDArray[Any]] = {}
        for name, parameters in self.parameters.items():
            values = np.asarray(columns[name], dtype=np.float64)
            clean = np.where(np.isfinite(values), values, parameters["missing"])
            clipped = np.clip(clean, parameters["lower"], parameters["upper"])
            output[name] = ((clipped - parameters["median"]) / parameters["scale"]).astype(
                np.float32
            )
        return output

    def manifest(self) -> dict[str, Any]:
        """Return fitted parameters and their canonical digest."""
        payload = {
            "schema_version": "owl.cadc.phase4-fold-transformer.v1",
            "clip_quantile": self.clip_quantile,
            "epsilon": self.epsilon,
            "parameters": self.parameters,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["digest"] = hashlib.sha256(encoded).hexdigest()
        return payload


class CanonicalDatasetBuilder:
    """Build a canonical dataset from verified factual and counterfactual catalogs."""

    def __init__(
        self,
        catalog: Phase4EvidenceCatalog,
        *,
        feature_registry: FeatureRegistry,
        outcome_registry: OutcomeRegistry,
        split_registry: SplitRegistry,
        backend: str,
        history_length: int = 8,
    ) -> None:
        if len(catalog.factual) != len(catalog.counterfactual):
            raise ValueError("factual/counterfactual corpus roots must be paired one-to-one")
        self.catalog = catalog
        self.feature_registry = feature_registry
        self.outcome_registry = outcome_registry
        self.split_registry = split_registry
        self.backend = backend
        self.reader = GPUParquetReader(backend)
        self.history_length = int(history_length)
        if self.history_length < 0:
            raise ValueError("history length cannot be negative")

    @property
    def dataset_id(self) -> str:
        """Return the stable identity of source and CADC-MORE 2 registries."""
        return stable_id(
            "canonical_dataset",
            self.catalog.catalog_id,
            self.feature_registry.digest,
            self.outcome_registry.digest,
            self.split_registry.digest,
        )

    def validate_split_rows(
        self, source_decision_ids: Any, split_roles: Any
    ) -> None:
        """Reject decision leakage and any materialized confirmatory role."""
        validate_no_leakage(np.asarray(source_decision_ids), np.asarray(split_roles))

    def write_partition(
        self,
        root: str | Path,
        name: str,
        columns: Mapping[str, Any],
        *,
        source_columns: Sequence[str],
        derivation: str,
    ) -> CanonicalPartitionReceipt:
        """Atomically write one normalized partition with provenance metadata."""
        output = Path(root) / name
        output.mkdir(parents=True, exist_ok=True)
        final = output / "part-000000.parquet"
        temporary = output / f".{final.name}.tmp.{os.getpid()}"
        rows = len(next(iter(columns.values()))) if columns else 0
        if any(len(value) != rows for value in columns.values()):
            raise ValueError(f"partition columns have unequal row counts: {name}")
        metadata = {
            "owl.cadc.phase4.dataset_schema": PHASE4_DATASET_SCHEMA_VERSION,
            "owl.cadc.phase4.dataset_id": self.dataset_id,
            "owl.cadc.phase3.source_sha256": self.catalog.provenance.phase3_source_sha256,
            "owl.cadc.phase4.feature_digest": self.feature_registry.digest,
            "owl.cadc.phase4.outcome_digest": self.outcome_registry.digest,
            "owl.cadc.phase4.split_digest": self.split_registry.digest,
        }
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("canonical dataset writing requires PyArrow") from exc
        table = pa.Table.from_pydict(dict(columns))
        encoded_metadata = {key.encode(): value.encode() for key, value in metadata.items()}
        table = table.replace_schema_metadata(encoded_metadata)
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        os.replace(temporary, final)
        return CanonicalPartitionReceipt(
            name=name,
            path=str(final),
            rows=rows,
            bytes=final.stat().st_size,
            sha256=sha256_file(final),
            source_columns=tuple(source_columns),
            derivation=derivation,
        )

    def build_spines(self, root: str | Path) -> tuple[CanonicalPartitionReceipt, ...]:
        """Stream normalized unit frames into canonical single-part Parquet tables."""

        import contextlib
        import gc

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("canonical dataset writing requires PyArrow") from exc

        names = (
            "decision_context",
            "candidate_context",
            "direction_context",
            "history_context",
            "branch_attempts",
            "branch_horizons",
            "branch_targets",
            "branch_events",
            "branch_contributions",
            "pair_labels",
            "survival_episodes",
            "information_episodes",
            "externality_targets",
            "nonexecutable_candidates",
        )
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "owl.cadc.phase4.dataset_schema": PHASE4_DATASET_SCHEMA_VERSION,
            "owl.cadc.phase4.dataset_id": self.dataset_id,
            "owl.cadc.phase3.source_sha256": self.catalog.provenance.phase3_source_sha256,
            "owl.cadc.phase4.feature_digest": self.feature_registry.digest,
            "owl.cadc.phase4.outcome_digest": self.outcome_registry.digest,
            "owl.cadc.phase4.split_digest": self.split_registry.digest,
        }
        encoded_metadata = {key.encode(): value.encode() for key, value in metadata.items()}
        writers: dict[str, Any] = {}
        schemas: dict[str, Any] = {}
        temporaries: dict[str, Path] = {}
        finals: dict[str, Path] = {}
        row_counts: dict[str, int] = dict.fromkeys(names, 0)
        source_columns: dict[str, tuple[str, ...]] = {}
        try:
            for factual, counterfactual in zip(
                self.catalog.factual, self.catalog.counterfactual, strict=True
            ):
                if self.backend == "cupy":
                    frames = self._build_gpu_unit(factual.table_paths, counterfactual)
                else:
                    frames = self._build_arrow_unit(factual.table_paths, counterfactual)
                try:
                    for name in names:
                        frame = frames[name]
                        columns = tuple(str(column) for column in frame.columns)
                        if name in source_columns and source_columns[name] != columns:
                            raise ValueError(f"canonical columns changed between units: {name}")
                        source_columns.setdefault(name, columns)
                        if self.backend == "cupy":
                            table = frame.to_arrow(preserve_index=False)
                        else:
                            table = frame.to_arrow()
                        table = table.replace_schema_metadata(encoded_metadata)
                        if name not in writers:
                            output = root_path / name
                            output.mkdir(parents=True, exist_ok=True)
                            final = output / "part-000000.parquet"
                            temporary = output / f".{final.name}.tmp.{os.getpid()}"
                            with contextlib.suppress(FileNotFoundError):
                                temporary.unlink()
                            schemas[name] = table.schema
                            temporaries[name] = temporary
                            finals[name] = final
                            writers[name] = pq.ParquetWriter(
                                temporary,
                                table.schema,
                                compression="zstd",
                                use_dictionary=True,
                            )
                        elif not table.schema.equals(schemas[name], check_metadata=True):
                            raise ValueError(
                                f"canonical Arrow schema changed between units: {name}"
                            )
                        writers[name].write_table(table, row_group_size=262_144)
                        row_counts[name] += int(table.num_rows)
                        del table
                finally:
                    frames.clear()
                    del frames
                    gc.collect()
                    if self.backend == "cupy":
                        import cupy as cp

                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
            for writer in writers.values():
                writer.close()
            writers.clear()
            receipts = []
            for name in names:
                temporary = temporaries[name]
                final = finals[name]
                os.replace(temporary, final)
                receipts.append(
                    CanonicalPartitionReceipt(
                        name=name,
                        path=str(final),
                        rows=row_counts[name],
                        bytes=final.stat().st_size,
                        sha256=sha256_file(final),
                        source_columns=source_columns[name],
                        derivation=f"normalized_join:{name}",
                    )
                )
            self.write_manifest(
                root_path / "manifests" / "dataset_manifest.json", receipts
            )
            return tuple(receipts)
        except Exception:
            for writer in writers.values():
                with contextlib.suppress(Exception):
                    writer.close()
            for temporary in temporaries.values():
                with contextlib.suppress(FileNotFoundError):
                    temporary.unlink()
            raise

    def _build_gpu_unit(
        self,
        factual_paths: Mapping[str, Path],
        counterfactual: Any,
    ) -> dict[str, Any]:
        import cudf

        factual = {
            name: cudf.read_parquet(str(path)) for name, path in factual_paths.items()
        }
        counter = {
            name: cudf.read_parquet([part.path for part in counterfactual.parts_for(name)])
            for name in (
                "source_decisions",
                "branch_attempts",
                "counterfactual_micro_rollouts",
                "branch_events",
                "branch_contributions",
                "candidate_pairs",
                "nonexecutable_candidates",
            )
        }
        source = counter["source_decisions"]
        join = list(SOURCE_JOIN_KEY)
        decisions = source.merge(
            factual["decisions"], on=join, how="left", suffixes=("_source", "")
        )
        if bool(decisions["run_id"].isna().any()):
            raise ValueError("GPU source/factual decision join is incomplete")
        base_key = list(DECISION_KEY)
        decisions = decisions.merge(
            factual["agent_context"], on=base_key, how="left", suffixes=("", "_agent")
        ).merge(
            factual["oracle_context"], on=base_key, how="left", suffixes=("", "_oracle")
        )
        if "execution" in factual:
            decisions = decisions.merge(
                factual["execution"],
                on=base_key,
                how="left",
                suffixes=("", "_execution"),
            )
        seed_value = int(decisions["seed"].iloc[0])
        split_role = self.split_registry.role_for_seed(seed_value).value
        decisions["split_role"] = split_role
        decisions["outer_fold"] = self.split_registry.outer_fold_for_seed(seed_value)
        provenance = decisions[
            [
                "source_decision_id",
                "run_id",
                "condition",
                "seed",
                "split_role",
                "outer_fold",
            ]
        ].drop_duplicates(subset=["source_decision_id"])
        if bool(provenance["source_decision_id"].duplicated().any()):
            raise ValueError("GPU source provenance key is not unique")
        for name in counter:
            if name != "source_decisions" and "source_decision_id" in counter[name].columns:
                before_rows = len(counter[name])
                merged = counter[name].merge(
                    provenance, on="source_decision_id", how="left", suffixes=("", "_source")
                )
                if len(merged) != before_rows:
                    raise ValueError(f"GPU provenance join changed row cardinality: {name}")
                if bool(merged["split_role"].isna().any()):
                    raise ValueError(f"GPU provenance join is incomplete: {name}")
                counter[name] = merged
        source_keys = source[[*join, "source_decision_id"]]
        candidate = source_keys.merge(factual["candidates"], on=join, how="left")
        direction = source_keys.merge(factual["action_directions"], on=join, how="left")
        candidate_counts = candidate.groupby("source_decision_id").size()
        direction_counts = direction.groupby("source_decision_id").size()
        if bool((candidate_counts != 22).any()) or bool((direction_counts != 16).any()):
            raise ValueError("GPU factual candidate/direction cardinality failed")
        information_evidence = None
        if "information" in factual:
            information_evidence = source_keys.merge(
                factual["information"], on=join, how="inner"
            )
            if "information_followups" in factual:
                followups = factual["information_followups"]
                information_evidence = information_evidence.merge(
                    followups,
                    left_on=[
                        "run_id",
                        "condition",
                        "seed",
                        "decision_sequence",
                        "ow_id",
                    ],
                    right_on=[
                        "run_id",
                        "condition",
                        "seed",
                        "source_decision_sequence",
                        "source_ow_id",
                    ],
                    how="left",
                    suffixes=("", "_followup"),
                )
        history = self._build_gpu_history(decisions, factual)
        source_lineage = decisions[
            ["source_decision_id", "lineage_id"]
        ].drop_duplicates(subset=["source_decision_id"]).rename(
            columns={"lineage_id": "source_lineage_id"}
        )
        targets, pairs, survival, information, externality = self._derive_gpu_targets(
            counter, information_evidence, source_lineage
        )
        return {
            "decision_context": decisions,
            "candidate_context": candidate,
            "direction_context": direction,
            "history_context": history,
            "branch_attempts": counter["branch_attempts"],
            "branch_horizons": counter["counterfactual_micro_rollouts"],
            "branch_targets": targets,
            "branch_events": counter["branch_events"],
            "branch_contributions": counter["branch_contributions"],
            "pair_labels": pairs,
            "survival_episodes": survival,
            "information_episodes": information,
            "externality_targets": externality,
            "nonexecutable_candidates": counter["nonexecutable_candidates"],
        }

    def _build_arrow_unit(
        self,
        factual_paths: Mapping[str, Path],
        counterfactual: Any,
    ) -> dict[str, Any]:
        try:
            import polars as pl
        except ImportError as exc:
            raise RuntimeError("CPU canonical ETL requires Polars") from exc
        factual = {name: pl.read_parquet(path) for name, path in factual_paths.items()}
        counter = {
            name: pl.read_parquet([part.path for part in counterfactual.parts_for(name)])
            for name in (
                "source_decisions",
                "branch_attempts",
                "counterfactual_micro_rollouts",
                "branch_events",
                "branch_contributions",
                "candidate_pairs",
                "nonexecutable_candidates",
            )
        }
        source = counter["source_decisions"]
        join = list(SOURCE_JOIN_KEY)
        decisions = source.join(factual["decisions"], on=join, how="left", suffix="_factual")
        if decisions["run_id"].null_count():
            raise ValueError("CPU source/factual decision join is incomplete")
        base_key = list(DECISION_KEY)
        decisions = decisions.join(
            factual["agent_context"], on=base_key, how="left", suffix="_agent"
        ).join(factual["oracle_context"], on=base_key, how="left", suffix="_oracle")
        if "execution" in factual:
            decisions = decisions.join(
                factual["execution"], on=base_key, how="left", suffix="_execution"
            )
        seed_value = int(decisions["seed"][0])
        split_role = self.split_registry.role_for_seed(seed_value).value
        outer_fold = self.split_registry.outer_fold_for_seed(seed_value)
        decisions = decisions.with_columns(
            pl.lit(split_role).alias("split_role"),
            pl.lit(outer_fold).alias("outer_fold"),
        )
        provenance = decisions.select(
            "source_decision_id",
            "run_id",
            "condition",
            "seed",
            "split_role",
            "outer_fold",
        ).unique(subset=["source_decision_id"])
        for name in counter:
            if name != "source_decisions" and "source_decision_id" in counter[name].columns:
                counter[name] = counter[name].join(
                    provenance, on="source_decision_id", how="left", suffix="_source"
                )
        source_keys = source.select(*join, "source_decision_id")
        candidate = source_keys.join(factual["candidates"], on=join, how="left")
        direction = source_keys.join(factual["action_directions"], on=join, how="left")
        if not (candidate.group_by("source_decision_id").len()["len"] == 22).all():
            raise ValueError("CPU candidate cardinality failed")
        if not (direction.group_by("source_decision_id").len()["len"] == 16).all():
            raise ValueError("CPU direction cardinality failed")
        information_evidence = None
        if "information" in factual:
            information_evidence = source_keys.join(
                factual["information"], on=join, how="inner"
            )
            if "information_followups" in factual:
                information_evidence = information_evidence.join(
                    factual["information_followups"],
                    left_on=[
                        "run_id",
                        "condition",
                        "seed",
                        "decision_sequence",
                        "ow_id",
                    ],
                    right_on=[
                        "run_id",
                        "condition",
                        "seed",
                        "source_decision_sequence",
                        "source_ow_id",
                    ],
                    how="left",
                    suffix="_followup",
                )
        history = self._build_polars_history(decisions, factual)
        source_lineage = decisions.select(
            "source_decision_id", pl.col("lineage_id").alias("source_lineage_id")
        ).unique(subset=["source_decision_id"])
        targets, pairs, survival, information, externality = self._derive_polars_targets(
            counter, information_evidence, source_lineage
        )
        return {
            "decision_context": decisions,
            "candidate_context": candidate,
            "direction_context": direction,
            "history_context": history,
            "branch_attempts": counter["branch_attempts"],
            "branch_horizons": counter["counterfactual_micro_rollouts"],
            "branch_targets": targets,
            "branch_events": counter["branch_events"],
            "branch_contributions": counter["branch_contributions"],
            "pair_labels": pairs,
            "survival_episodes": survival,
            "information_episodes": information,
            "externality_targets": externality,
            "nonexecutable_candidates": counter["nonexecutable_candidates"],
        }

    def _build_gpu_history(self, decisions: Any, factual: Mapping[str, Any]) -> Any:
        """Build a prior-tick agent history without crossing run/lineage boundaries."""

        import cudf

        source_columns = [
            "source_decision_id",
            "run_id",
            "condition",
            "seed",
            "tick",
            "ow_id",
            "lineage_id",
        ]
        source = decisions[source_columns].drop_duplicates(
            subset=["source_decision_id"]
        ).rename(columns={"tick": "source_tick"})
        base = list(DECISION_KEY)
        historical = factual["decisions"][
            [*base, "lineage_id"]
        ].merge(factual["agent_context"], on=base, how="inner").rename(
            columns={"tick": "history_tick"}
        )
        join = ["run_id", "condition", "seed", "ow_id", "lineage_id"]
        history = source.merge(historical, on=join, how="inner")
        history = history[history["history_tick"] < history["source_tick"]]
        history = history.sort_values(
            ["source_decision_id", "history_tick", "decision_sequence"],
            ascending=[True, False, False],
        )
        history["history_lag"] = history.groupby("source_decision_id").cumcount()
        history = history[history["history_lag"] < self.history_length]
        if not isinstance(history, cudf.DataFrame):
            raise TypeError("GPU history unexpectedly left the cuDF backend")
        return history

    def _build_polars_history(self, decisions: Any, factual: Mapping[str, Any]) -> Any:
        """Build the same bounded history with a Polars reference path."""

        import polars as pl

        source = decisions.select(
            "source_decision_id",
            "run_id",
            "condition",
            "seed",
            pl.col("tick").alias("source_tick"),
            "ow_id",
            "lineage_id",
        ).unique(subset=["source_decision_id"])
        base = list(DECISION_KEY)
        historical = factual["decisions"].select(
            *base, "lineage_id"
        ).join(factual["agent_context"], on=base, how="inner").rename(
            {"tick": "history_tick"}
        )
        join = ["run_id", "condition", "seed", "ow_id", "lineage_id"]
        return source.join(historical, on=join, how="inner").filter(
            pl.col("history_tick") < pl.col("source_tick")
        ).sort(
            "source_decision_id",
            "history_tick",
            "decision_sequence",
            descending=[False, True, True],
        ).with_columns(
            pl.col("source_decision_id")
            .cum_count()
            .over("source_decision_id")
            .sub(1)
            .alias("history_lag")
        ).filter(pl.col("history_lag") < self.history_length)

    def _derive_gpu_targets(
        self,
        counter: Mapping[str, Any],
        information_evidence: Any | None,
        source_lineage: Any,
    ) -> tuple[Any, ...]:

        horizon = counter["counterfactual_micro_rollouts"].copy(deep=True).merge(
            source_lineage, on="source_decision_id", how="left"
        )
        if bool(horizon["source_lineage_id"].isna().any()):
            raise ValueError("GPU source-lineage join is incomplete")
        horizon["focal_lineage_persistence"] = horizon["alive"].astype("bool") & (
            horizon["lineage_id"] == horizon["source_lineage_id"]
        )
        horizon["death_by_horizon"] = ~horizon["alive"].astype("bool")
        horizon["homeostatic_proxy"] = (
            horizon["health_delta"]
            + 0.7 * horizon["resource_delta"]
            + 0.3 * horizon["boundary_delta"]
            + 0.3 * horizon["integration_delta"]
            + 0.2 * horizon["memory_delta"]
        )
        horizon["agent_risk_neutral"] = horizon["homeostatic_proxy"]
        horizon["agent_risk_averse"] = (
            horizon["homeostatic_proxy"] - 4.0 * horizon["death_by_horizon"]
        )
        horizon = self._attach_gpu_death_cause(horizon, counter["branch_events"])
        pairs = counter["candidate_pairs"].copy(deep=True)
        label_columns = [
            "branch_id",
            "horizon",
            "agent_risk_neutral",
            "agent_risk_averse",
        ]
        left = horizon[label_columns].rename(
            columns={
                "branch_id": "branch_a",
                "agent_risk_neutral": "value_a_risk_neutral",
                "agent_risk_averse": "value_a_risk_averse",
            }
        )
        right = horizon[label_columns].rename(
            columns={
                "branch_id": "branch_b",
                "agent_risk_neutral": "value_b_risk_neutral",
                "agent_risk_averse": "value_b_risk_averse",
            }
        )
        pairs = pairs.merge(left, on=["branch_a", "horizon"], how="left").merge(
            right, on=["branch_b", "horizon"], how="left"
        )
        pairs["advantage_risk_neutral"] = (
            pairs["value_a_risk_neutral"] - pairs["value_b_risk_neutral"]
        )
        pairs["advantage_risk_averse"] = (
            pairs["value_a_risk_averse"] - pairs["value_b_risk_averse"]
        )
        pairs["win_label"] = (
            (pairs["advantage_risk_averse"] > 1e-6).astype("float32")
            + 0.5 * (pairs["advantage_risk_averse"].abs() <= 1e-6).astype("float32")
        )
        pairs["magnitude"] = pairs["advantage_risk_averse"].abs()
        survival = horizon[
            [
                "branch_id",
                "source_decision_id",
                "forced_action",
                "repeat_index",
                "horizon",
                "alive",
                "first_death_tick",
                "death_evidence",
                "horizon_status",
            ]
        ].copy(deep=True)
        survival["event_observed"] = (~survival["alive"].astype("bool")) & (
            survival["first_death_tick"] >= 0
        )
        survival["event_time"] = survival["first_death_tick"].where(
            survival["event_observed"], survival["horizon"]
        )
        information = horizon[horizon["forced_action"].isin([1, 11])][
            [
                "branch_id",
                "source_decision_id",
                "forced_action",
                "repeat_index",
                "horizon",
                "active_sense_new_cell_count",
                "active_sense_new_target_count",
                "memory_delta",
                "resource_delta",
                "agent_risk_neutral",
            ]
        ]
        if information_evidence is not None:
            information = information.merge(
                information_evidence, on="source_decision_id", how="left", suffixes=("", "_factual")
            )
        if "followup_status" not in information:
            information["followup_status"] = -1
        information["episode_censored"] = information["followup_status"].fillna(-1) != 1
        anchor = counter["branch_attempts"][counter["branch_attempts"]["selected_anchor"]][
            ["source_decision_id", "repeat_index", "branch_id"]
        ].rename(columns={"branch_id": "anchor_branch_id"})
        anchor_horizon = anchor.merge(
            horizon[
                [
                    "branch_id",
                    "horizon",
                    "population",
                    "world_food",
                    "world_toxin",
                    "world_waste",
                    "focal_lineage_persistence",
                ]
            ].rename(columns={"branch_id": "anchor_branch_id"}),
            on="anchor_branch_id",
            how="inner",
        ).rename(
            columns={
                "population": "anchor_population",
                "world_food": "anchor_world_food",
                "world_toxin": "anchor_world_toxin",
                "world_waste": "anchor_world_waste",
                "focal_lineage_persistence": "anchor_focal_lineage_persistence",
            }
        )
        externality = horizon.merge(
            anchor_horizon,
            on=["source_decision_id", "repeat_index", "horizon"],
            how="left",
        )
        for field in (
            "population",
            "world_food",
            "world_toxin",
            "world_waste",
            "focal_lineage_persistence",
        ):
            externality[f"{field}_delta_vs_anchor"] = (
                externality[field] - externality[f"anchor_{field}"]
            )
        keep = [
            "branch_id",
            "source_decision_id",
            "forced_action",
            "repeat_index",
            "horizon",
            "population_delta_vs_anchor",
            "world_food_delta_vs_anchor",
            "world_toxin_delta_vs_anchor",
            "world_waste_delta_vs_anchor",
            "focal_lineage_persistence_delta_vs_anchor",
        ]
        return horizon, pairs, survival, information, externality[keep]

    @staticmethod
    def _attach_gpu_death_cause(horizon: Any, events: Any) -> Any:
        """Attach cause classes with a device-native branch/horizon event reduction."""

        import cupy as cp

        keys = horizon[["branch_id", "horizon"]].drop_duplicates()
        expanded = keys.merge(
            events[["branch_id", "branch_tick", "event_code"]],
            on="branch_id",
            how="left",
        )
        expanded = expanded[expanded["branch_tick"] <= expanded["horizon"]]
        for name, event_code in (
            ("starvation_evidence", CADCEventCode.STARVATION_EVIDENCE),
            ("toxin_evidence", CADCEventCode.TOXIN_DAMAGE_EVIDENCE),
            ("death_event", CADCEventCode.DEATH),
        ):
            expanded[name] = expanded["event_code"] == int(event_code)
        flags = expanded.groupby(["branch_id", "horizon"], as_index=False).agg(
            {
                "starvation_evidence": "max",
                "toxin_evidence": "max",
                "death_event": "max",
            }
        )
        horizon = horizon.merge(flags, on=["branch_id", "horizon"], how="left")
        for name in ("starvation_evidence", "toxin_evidence", "death_event"):
            horizon[name] = horizon[name].fillna(False).astype("bool")
        alive = horizon["alive"].values.astype(cp.bool_)
        starvation = horizon["starvation_evidence"].values.astype(cp.bool_)
        toxin = horizon["toxin_evidence"].values.astype(cp.bool_)
        absent = horizon["horizon_status"].values == 2
        horizon["death_cause"] = cp.where(
            alive,
            0,
            cp.where(
                absent | (starvation & toxin),
                4,
                cp.where(starvation, 1, cp.where(toxin, 2, 3)),
            ),
        ).astype(cp.int8)
        for cause_code in range(5):
            horizon[f"death_cause_{cause_code}"] = horizon["death_cause"] == cause_code
        return horizon

    def _derive_polars_targets(
        self,
        counter: Mapping[str, Any],
        information_evidence: Any | None,
        source_lineage: Any,
    ) -> tuple[Any, ...]:
        import polars as pl

        horizon = counter["counterfactual_micro_rollouts"].join(
            source_lineage, on="source_decision_id", how="left", validate="m:1"
        )
        if horizon["source_lineage_id"].null_count():
            raise ValueError("CPU source-lineage join is incomplete")
        horizon = horizon.with_columns(
            (~pl.col("alive")).alias("death_by_horizon"),
            (
                pl.col("alive")
                & (pl.col("lineage_id") == pl.col("source_lineage_id"))
            ).alias("focal_lineage_persistence"),
            (
                pl.col("health_delta")
                + 0.7 * pl.col("resource_delta")
                + 0.3 * pl.col("boundary_delta")
                + 0.3 * pl.col("integration_delta")
                + 0.2 * pl.col("memory_delta")
            ).alias("homeostatic_proxy"),
        ).with_columns(
            pl.col("homeostatic_proxy").alias("agent_risk_neutral"),
            (
                pl.col("homeostatic_proxy")
                - 4.0 * pl.col("death_by_horizon").cast(pl.Float32)
            ).alias("agent_risk_averse"),
        )
        horizon = self._attach_polars_death_cause(horizon, counter["branch_events"])
        label_columns = [
            "branch_id",
            "horizon",
            "agent_risk_neutral",
            "agent_risk_averse",
        ]
        left = horizon.select(label_columns).rename(
            {
                "branch_id": "branch_a",
                "agent_risk_neutral": "value_a_risk_neutral",
                "agent_risk_averse": "value_a_risk_averse",
            }
        )
        right = horizon.select(label_columns).rename(
            {
                "branch_id": "branch_b",
                "agent_risk_neutral": "value_b_risk_neutral",
                "agent_risk_averse": "value_b_risk_averse",
            }
        )
        pairs = counter["candidate_pairs"].join(
            left, on=["branch_a", "horizon"], how="left"
        ).join(right, on=["branch_b", "horizon"], how="left").with_columns(
            (pl.col("value_a_risk_neutral") - pl.col("value_b_risk_neutral")).alias(
                "advantage_risk_neutral"
            ),
            (pl.col("value_a_risk_averse") - pl.col("value_b_risk_averse")).alias(
                "advantage_risk_averse"
            ),
        ).with_columns(
            pl.when(pl.col("advantage_risk_averse") > 1e-6)
            .then(1.0)
            .when(pl.col("advantage_risk_averse").abs() <= 1e-6)
            .then(0.5)
            .otherwise(0.0)
            .alias("win_label"),
            pl.col("advantage_risk_averse").abs().alias("magnitude"),
        )
        survival = horizon.select(
            "branch_id",
            "source_decision_id",
            "forced_action",
            "repeat_index",
            "horizon",
            "alive",
            "first_death_tick",
            "death_evidence",
            "horizon_status",
        ).with_columns(
            ((~pl.col("alive")) & (pl.col("first_death_tick") >= 0)).alias(
                "event_observed"
            )
        ).with_columns(
            pl.when(pl.col("event_observed"))
            .then(pl.col("first_death_tick"))
            .otherwise(pl.col("horizon"))
            .alias("event_time")
        )
        information = horizon.filter(pl.col("forced_action").is_in([1, 11])).select(
            "branch_id",
            "source_decision_id",
            "forced_action",
            "repeat_index",
            "horizon",
            "active_sense_new_cell_count",
            "active_sense_new_target_count",
            "memory_delta",
            "resource_delta",
            "agent_risk_neutral",
        )
        if information_evidence is not None:
            information = information.join(
                information_evidence, on="source_decision_id", how="left", suffix="_factual"
            )
        if "followup_status" not in information.columns:
            information = information.with_columns(pl.lit(-1).alias("followup_status"))
        information = information.with_columns(
            (pl.col("followup_status").fill_null(-1) != 1).alias("episode_censored")
        )
        anchor = counter["branch_attempts"].filter(pl.col("selected_anchor")).select(
            "source_decision_id", "repeat_index", "branch_id"
        ).rename({"branch_id": "anchor_branch_id"})
        anchor_horizon = anchor.join(
            horizon.select(
                pl.col("branch_id").alias("anchor_branch_id"),
                "horizon",
                pl.col("population").alias("anchor_population"),
                pl.col("world_food").alias("anchor_world_food"),
                pl.col("world_toxin").alias("anchor_world_toxin"),
                pl.col("world_waste").alias("anchor_world_waste"),
                pl.col("focal_lineage_persistence").alias(
                    "anchor_focal_lineage_persistence"
                ),
            ),
            on="anchor_branch_id",
            how="inner",
        )
        externality = horizon.join(
            anchor_horizon,
            on=["source_decision_id", "repeat_index", "horizon"],
            how="left",
        ).with_columns(
            *[
                (pl.col(field) - pl.col(f"anchor_{field}")).alias(
                    f"{field}_delta_vs_anchor"
                )
                for field in (
                    "population",
                    "world_food",
                    "world_toxin",
                    "world_waste",
                    "focal_lineage_persistence",
                )
            ]
        ).select(
            "branch_id",
            "source_decision_id",
            "forced_action",
            "repeat_index",
            "horizon",
            "population_delta_vs_anchor",
            "world_food_delta_vs_anchor",
            "world_toxin_delta_vs_anchor",
            "world_waste_delta_vs_anchor",
            "focal_lineage_persistence_delta_vs_anchor",
        )
        return horizon, pairs, survival, information, externality

    @staticmethod
    def _attach_polars_death_cause(horizon: Any, events: Any) -> Any:
        """Attach the same cause classes through a Polars columnar reduction."""

        import polars as pl

        keys = horizon.select("branch_id", "horizon").unique()
        flags = keys.join(
            events.select("branch_id", "branch_tick", "event_code"),
            on="branch_id",
            how="left",
        ).filter(pl.col("branch_tick") <= pl.col("horizon")).group_by(
            "branch_id", "horizon"
        ).agg(
            (pl.col("event_code") == int(CADCEventCode.STARVATION_EVIDENCE))
            .any()
            .alias("starvation_evidence"),
            (pl.col("event_code") == int(CADCEventCode.TOXIN_DAMAGE_EVIDENCE))
            .any()
            .alias("toxin_evidence"),
            (pl.col("event_code") == int(CADCEventCode.DEATH))
            .any()
            .alias("death_event"),
        )
        result = horizon.join(
            flags, on=["branch_id", "horizon"], how="left", validate="m:1"
        ).with_columns(
            pl.col("starvation_evidence").fill_null(False),
            pl.col("toxin_evidence").fill_null(False),
            pl.col("death_event").fill_null(False),
        ).with_columns(
            pl.when(pl.col("alive"))
            .then(0)
            .when(
                (pl.col("horizon_status") == 2)
                | (pl.col("starvation_evidence") & pl.col("toxin_evidence"))
            )
            .then(4)
            .when(pl.col("starvation_evidence"))
            .then(1)
            .when(pl.col("toxin_evidence"))
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("death_cause")
        )
        return result.with_columns(
            *[
                (pl.col("death_cause") == code)
                .cast(pl.Float32)
                .alias(f"death_cause_{code}")
                for code in range(5)
            ]
        )

    def _concat_frames(self, frames: Sequence[Any]) -> Any:
        if self.backend == "cupy":
            import cudf

            return cudf.concat(list(frames), ignore_index=True)
        import polars as pl

        return pl.concat(list(frames), how="diagonal_relaxed", rechunk=True)

    def _write_frame(
        self, root: str | Path, name: str, frame: Any, *, derivation: str
    ) -> CanonicalPartitionReceipt:
        output = Path(root) / name
        output.mkdir(parents=True, exist_ok=True)
        final = output / "part-000000.parquet"
        temporary = output / f".{final.name}.tmp.{os.getpid()}"
        if self.backend == "cupy":
            frame.to_parquet(temporary, compression="zstd", index=False)
            columns = tuple(str(value) for value in frame.columns)
            rows = len(frame)
        else:
            frame.write_parquet(temporary, compression="zstd", statistics=True)
            columns = tuple(str(value) for value in frame.columns)
            rows = frame.height
        os.replace(temporary, final)
        metadata = {
            "schema_version": "owl.cadc.phase4-partition-metadata.v1",
            "dataset_id": self.dataset_id,
            "source_sha256": self.catalog.provenance.phase3_source_sha256,
            "feature_registry_digest": self.feature_registry.digest,
            "outcome_registry_digest": self.outcome_registry.digest,
            "split_registry_digest": self.split_registry.digest,
            "derivation": derivation,
            "columns": list(columns),
            "rows": rows,
            "sha256": sha256_file(final),
        }
        atomic_json(output / "_metadata.json", metadata)
        return CanonicalPartitionReceipt(
            name=name,
            path=str(final),
            rows=rows,
            bytes=final.stat().st_size,
            sha256=str(metadata["sha256"]),
            source_columns=columns,
            derivation=derivation,
        )

    def write_manifest(
        self, path: str | Path, receipts: Sequence[CanonicalPartitionReceipt]
    ) -> None:
        """Write the source-bound canonical dataset manifest."""
        atomic_json(
            path,
            {
                "schema_version": PHASE4_DATASET_SCHEMA_VERSION,
                "dataset_id": self.dataset_id,
                "provenance": self.catalog.provenance.to_dict(),
                "feature_registry_digest": self.feature_registry.digest,
                "outcome_registry_digest": self.outcome_registry.digest,
                "split_registry_digest": self.split_registry.digest,
                "parts": [asdict(receipt) for receipt in receipts],
                "phase5_locked": True,
                "passed": True,
            },
        )
