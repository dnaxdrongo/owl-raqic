"""Action-agnostic viability baselines retained as mandatory comparators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from owl.cadc.models._optional import TorchModule, require_torch


class ActionAgnosticBaseline(TorchModule):
    """Neural baseline that cannot receive action or target features."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        _, nn = require_torch()
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, context: Any) -> Any:
        """Predict viability from agent context without action information."""
        if context.ndim != 2:
            raise ValueError("action-agnostic baseline accepts context [B,F] only")
        return self.network(context)


class XGBoostActionAgnosticBaseline:
    """GPU histogram XGBoost baseline with deterministic source/config metadata."""

    def __init__(self, *, seed: int, device: str = "cuda") -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("XGBoost device must be cpu or cuda")
        self.seed = int(seed)
        self.device = device
        self.booster: Any | None = None
        self.feature_names: tuple[str, ...] = ()

    def fit(
        self,
        features: npt.NDArray[Any],
        targets: npt.NDArray[Any],
        *,
        feature_names: tuple[str, ...],
        rounds: int,
    ) -> None:
        """Fit the action-agnostic GPU/CPU histogram baseline."""
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("XGBoost baseline requires the cadc training extra") from exc
        if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.size:
            raise ValueError("XGBoost feature/target shapes are incompatible")
        if features.shape[1] != len(feature_names):
            raise ValueError("XGBoost feature names do not match feature width")
        matrix = xgb.DMatrix(features, label=targets, feature_names=list(feature_names))
        parameters = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": self.device,
            "seed": self.seed,
            "nthread": 1,
        }
        self.booster = xgb.train(parameters, matrix, num_boost_round=rounds)
        self.feature_names = feature_names

    def predict(self, features: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict outcomes from a fitted action-agnostic booster."""
        if self.booster is None:
            raise RuntimeError("XGBoost baseline has not been fit")
        import xgboost as xgb

        matrix = xgb.DMatrix(features, feature_names=list(self.feature_names))
        return np.asarray(self.booster.predict(matrix))

    def save(self, path: str | Path) -> None:
        """Persist the fitted booster using XGBoost's model format."""
        if self.booster is None:
            raise RuntimeError("XGBoost baseline has not been fit")
        self.booster.save_model(str(path))
