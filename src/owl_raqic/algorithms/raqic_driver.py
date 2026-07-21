from __future__ import annotations

import numpy as np

from owl_raqic.algorithms.feature_pipeline import compute_action_phases, compute_scores
from owl_raqic.algorithms.sampling import counts_from_probabilities
from owl_raqic.config import RAQICAlgorithmConfig
from owl_raqic.math.checks import (
    check_density_matrix,
    check_kraus_completeness,
    check_state_normalization,
    check_top_down_bias,
)
from owl_raqic.math.instruments import (
    action_amplitudes,
    preparation_kraus_from_amplitudes,
    simulate_recursive_ensemble,
)
from owl_raqic.math.intentions import normalize_intention
from owl_raqic.random_contract import RNGStream, categorical
from owl_raqic.types import RAQICActionSet, RAQICDecisionResult, RAQICFeaturePacket


class RAQICDecisionEngine:
    """Standalone future-OWL-ready decision engine.

    It does not import or mutate OWL. It accepts feature packets and returns action distributions.
    """

    def __init__(
        self, config: RAQICAlgorithmConfig | None = None, action_set: RAQICActionSet | None = None
    ):
        self.config = config or RAQICAlgorithmConfig()
        self.action_set = action_set or RAQICActionSet()
        if self.config.registers.n_actions > len(self.action_set):
            raise ValueError("config n_actions exceeds action set length")

    def decide(self, packet: RAQICFeaturePacket, sample: bool = False) -> RAQICDecisionResult:
        scores = compute_scores(packet, self.config, self.action_set.names)
        phases = compute_action_phases(packet, self.config)
        mask = packet.authority_mask
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)[: self.config.registers.n_actions]
        intention = (
            None
            if packet.parent_intention is None
            else normalize_intention(
                packet.parent_intention[: self.config.registers.n_actions],
                self.config.registers.n_actions,
            )
        )
        baseline_amps, baseline_probs = action_amplitudes(
            scores, phases * 0, None, 0.0, self.config.action_temperature, mask
        )
        amps, probs = action_amplitudes(
            scores,
            phases,
            intention,
            self.config.beta_intention,
            self.config.action_temperature,
            mask,
        )
        kraus, Uprep, projectors = preparation_kraus_from_amplitudes(amps)
        checks = {
            "state_normalized": check_state_normalization(amps),
            "kraus": check_kraus_completeness(kraus),
            "probabilities_sum": float(probs.sum()),
            "top_down_bias": None,
        }
        if intention is not None:
            target = int(np.argmax(intention))
            checks["top_down_bias"] = check_top_down_bias(baseline_probs, probs, target)
        sampled = (
            int(
                categorical(
                    probs[None, :],
                    self.config.seed,
                    packet.tick,
                    np.asarray([packet.ow_id], dtype=np.uint64),
                    RNGStream.RAQIC_READOUT,
                    xp=np,
                )[0]
            )
            if sample
            else None
        )
        counts = (
            counts_from_probabilities(probs, self.config.shots, self.config.seed)
            if self.config.mode in ("dynamic", "walk")
            else None
        )
        rec = simulate_recursive_ensemble(amps, rounds=max(1, self.config.rounds))
        checks["recursive_trace_preserved"] = bool(np.allclose(rec["traces"], 1.0, atol=1e-10))
        checks["recursive_positive"] = bool(np.all(rec["min_eigenvalues"] >= -1e-10))
        checks["final_density"] = check_density_matrix(rec["rho"])
        action_name = None if sampled is None else self.action_set.names[sampled]
        return RAQICDecisionResult(
            ow_id=packet.ow_id,
            scale_id=packet.scale_id,
            tick=packet.tick,
            action_probabilities=probs,
            sampled_action=sampled,
            sampled_action_name=action_name,
            measurement_record={
                "mode": self.config.mode,
                "scores": scores.tolist(),
                "phases": phases.tolist(),
                "counts": counts,
                "recursive_traces": rec["traces"].tolist(),
            },
            simulator_metadata={
                "engine": "cpu_audit_reference",
                "finite_projection": {
                    "active_places": ["infinity", *list(self.config.active_places.primes)],
                    "approximation_status": "exact_within_selected_finite_model",
                    "approximation_sources": [
                        "finite active-place set",
                        "feature bins",
                        "finite action basis",
                    ],
                },
                "not_integrated_with_owl": True,
            },
            recovery_checks=checks,
        )

    def decide_batch(
        self, packets: list[RAQICFeaturePacket], sample: bool = False
    ) -> list[RAQICDecisionResult]:
        return [self.decide(p, sample=sample) for p in packets]
