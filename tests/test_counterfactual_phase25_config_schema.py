from __future__ import annotations

import pytest
from pydantic import ValidationError

from owl.core.config import SimulationConfig, load_config


def test_counterfactual_defaults_are_inert_and_target_numpy_fails() -> None:
    assert not SimulationConfig().counterfactual.enabled
    cfg = load_config("configs/cadc_phase3_phase25_numpy_smoke.yaml")
    data = cfg.model_dump(mode="json")
    data["counterfactual"]["branch_execution_mode"] = "target_gpu_required"
    with pytest.raises(ValidationError, match="cannot use NumPy"):
        SimulationConfig.model_validate(data)
