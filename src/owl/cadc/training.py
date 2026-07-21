"""Cross-fit, ensemble, checkpoint, and training-ledger orchestration."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from owl.cadc.artifacts import atomic_json, receipt_for, write_receipt
from owl.cadc.models._optional import require_torch
from owl.cadc.schema import PHASE4_MODEL_ARTIFACT_VERSION, stable_id


@dataclass(frozen=True)
class TrainingLedgerRow:
    """One finite, provenance-bound training progress observation."""
    training_run_id: str
    member_seed: int
    epoch: int
    step: int
    train_loss: float
    validation_loss: float
    learning_rate: float
    gradient_norm: float
    nonfinite_count: int
    examples_per_second: float
    gpu_memory_bytes: int


@dataclass
class TrainingRun:
    """One provenance-bound training run with fail-closed metric logging."""

    source_sha256: str
    dataset_sha256: str
    split_sha256: str
    config_sha256: str
    member_seed: int
    ledger: list[TrainingLedgerRow] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        """Return the stable identity of this source/data/split/member run."""
        return stable_id(
            "training_run",
            self.source_sha256,
            self.dataset_sha256,
            self.split_sha256,
            self.config_sha256,
            self.member_seed,
        )

    def record(
        self,
        *,
        epoch: int,
        step: int,
        train_loss: float,
        validation_loss: float,
        learning_rate: float,
        gradient_norm: float,
        nonfinite_count: int,
        examples_per_second: float,
        gpu_memory_bytes: int,
    ) -> None:
        """Append one finite metric row or fail closed on numerical corruption."""
        numeric = (train_loss, validation_loss, learning_rate, gradient_norm)
        if not all(math.isfinite(value) for value in numeric) or nonfinite_count:
            raise FloatingPointError("nonfinite loss/gradient encountered during Phase 4 training")
        self.ledger.append(
            TrainingLedgerRow(
                self.run_id,
                self.member_seed,
                epoch,
                step,
                train_loss,
                validation_loss,
                learning_rate,
                gradient_norm,
                nonfinite_count,
                examples_per_second,
                gpu_memory_bytes,
            )
        )

    def write(self, path: str | Path) -> None:
        """Atomically persist the complete member training ledger."""
        atomic_json(
            path,
            {
                "schema_version": "owl.cadc.phase4-training-ledger.v1",
                "training_run_id": self.run_id,
                "source_sha256": self.source_sha256,
                "dataset_sha256": self.dataset_sha256,
                "split_sha256": self.split_sha256,
                "config_sha256": self.config_sha256,
                "member_seed": self.member_seed,
                "rows": [asdict(value) for value in self.ledger],
            },
        )


@dataclass(frozen=True)
class FoldDefinition:
    """Explicit train, validation, and test groups for one outer fold."""
    outer_fold: int
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


class CrossFitCoordinator:
    """Create leave-group-out outer folds without exposing outer tests to selection."""

    def __init__(self, group_ids: Sequence[str], outer_folds: Sequence[int]) -> None:
        groups = np.asarray(group_ids).astype(str)
        folds = np.asarray(outer_folds, dtype=np.int64)
        if groups.shape != folds.shape:
            raise ValueError("cross-fit groups and folds do not align")
        self.groups = groups
        self.folds = folds

    def definitions(self) -> tuple[FoldDefinition, ...]:
        """Build nonempty leave-group-out fold definitions."""
        values: list[FoldDefinition] = []
        unique = np.unique(self.folds)
        for outer in unique:
            test = np.unique(self.groups[self.folds == outer])
            remaining = np.unique(self.groups[self.folds != outer])
            selector = np.asarray(
                [
                    int.from_bytes(value.encode()[:8].ljust(8, b"\0"), "big") % 5
                    for value in remaining
                ]
            )
            validation = remaining[selector == 0]
            train = remaining[selector != 0]
            if not train.size or not validation.size or not test.size:
                raise ValueError(f"cross-fit fold {outer} has an empty partition")
            values.append(
                FoldDefinition(
                    outer_fold=int(outer),
                    train_groups=tuple(train.tolist()),
                    validation_groups=tuple(validation.tolist()),
                    test_groups=tuple(test.tolist()),
                )
            )
        return tuple(values)


class EnsembleTrainer:
    """Train exactly the configured member set and persist every member receipt."""

    def __init__(
        self,
        model_factory: Callable[[int], Any],
        *,
        member_seeds: Sequence[int],
        device: str,
    ) -> None:
        if not member_seeds or len(set(member_seeds)) != len(member_seeds):
            raise ValueError("ensemble member seeds must be nonempty and unique")
        self.model_factory = model_factory
        self.member_seeds = tuple(int(value) for value in member_seeds)
        self.device = device

    def build_members(self) -> list[Any]:
        """Construct exactly the configured deterministic ensemble members."""
        torch, _ = require_torch()
        members = []
        for seed in self.member_seeds:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            members.append(self.model_factory(seed).to(self.device))
        return members

    def save_members(
        self,
        members: Sequence[Any],
        root: str | Path,
        *,
        source_sha256: str,
        config_sha256: str,
    ) -> tuple[dict[str, Any], ...]:
        """Persist every configured member and its checksum receipt."""
        torch, _ = require_torch()
        if len(members) != len(self.member_seeds):
            raise RuntimeError("refusing to persist a partial ensemble")
        output = Path(root)
        output.mkdir(parents=True, exist_ok=True)
        receipts = []
        for seed, model in zip(self.member_seeds, members, strict=True):
            path = output / f"member-{seed}.pt"
            temporary = output / f".{path.name}.tmp"
            torch.save(model.state_dict(), temporary)
            temporary.replace(path)
            receipt = receipt_for(
                path,
                schema_version=PHASE4_MODEL_ARTIFACT_VERSION,
                source_sha256=source_sha256,
                config_sha256=config_sha256,
            )
            write_receipt(output / f"member-{seed}.receipt.json", receipt)
            receipts.append(asdict(receipt))
        return tuple(receipts)


def train_epoch(
    model: Any,
    batches: Iterable[Mapping[str, Any]],
    *,
    optimizer: Any,
    loss_function: Callable[[Any, Mapping[str, Any]], Any],
    gradient_clip: float,
) -> tuple[float, float, int, float]:
    """Run one fail-closed epoch and return loss, gradient norm, rows, throughput."""
    torch, _ = require_torch()
    model.train()
    total_loss = 0.0
    total_rows = 0
    maximum_gradient = 0.0
    started = time.perf_counter()
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch["inputs"])
        loss = loss_function(output, batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite Phase 4 training loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if not bool(torch.isfinite(gradient)):
            raise FloatingPointError("nonfinite Phase 4 gradient norm")
        optimizer.step()
        rows = int(batch["rows"])
        total_rows += rows
        total_loss += float(loss.detach()) * rows
        maximum_gradient = max(maximum_gradient, float(gradient.detach()))
    elapsed = max(time.perf_counter() - started, 1e-12)
    if not total_rows:
        raise ValueError("training epoch received no rows")
    return total_loss / total_rows, maximum_gradient, total_rows, total_rows / elapsed
