# Observer-Window Life and RAQIC

This repository contains my final implementation of Observer-Window Life (OWL), an
array-first artificial-life simulation in which bounded observer windows sense local
conditions, maintain internal state, coordinate across scales, and select actions through
the Recursive Adelic Quantum-Inspired Circuit (RAQIC). I designed the project to study how
local viability, information, coherence, and hierarchical context influence adaptive
decisions in a controlled computational ecology.

The implementation is a scientific simulation, not evidence that the simulated entities
are conscious and not a claim of biological quantum computation. The quantum circuit is a
validation representation of a finite decision law; the production simulator uses dense
CPU or GPU mathematics and makes no quantum-advantage claim.

## Research design

The experiment compares matched simulation conditions that selectively enable the RAQIC
utility, resonance, and interference mechanisms. Observer state is represented at local,
patch, and global aggregations of one continuous simulated system. The final CADC pipeline
uses factual recorder data and isolated, forced-action counterfactual branches to estimate
multi-horizon outcomes, rank executable candidates, quantify uncertainty, and abstain when
an action lies outside observed support.

The multiscale observer-window framing is informed by the Nested Observer Windows model
[Riddle and Schooler, 2024](https://doi.org/10.1093/nc/niae010). The finite p-adic and
adelic construction uses established mathematical background without claiming a biological
adelic mechanism [Gouvêa, 1997](https://doi.org/10.1007/978-3-642-59058-0). RAQIC's circuit
validation follows standard quantum-information and uniformly controlled gate methods
[Nielsen and Chuang, 2010](https://doi.org/10.1017/CBO9780511976667) and
[Bergholm et al., 2005](https://doi.org/10.1103/PhysRevA.71.052330). The complete scholarly
library and claim guardrails are in `docs/REFERENCES.md` and
`docs/REFERENCES.json`.

## Pipeline

1. `owl.core` validates configuration and initializes authoritative state.
2. `owl.engine` provides the readable CPU/reference tick implementation.
3. `owl.gpu` executes persistent, vectorized scientific stages on supported CUDA hardware.
4. `owl.raqic` and `owl_raqic` calculate and validate the RAQIC decision distribution.
5. `owl.record` writes factual, columnar evidence without changing simulation state.
6. `owl.counterfactual` clones checkpoints and evaluates forced actions under paired random
   streams.
7. `owl.cadc` builds leakage-controlled features, outcomes, models, calibration, support,
   evaluation, and inference artifacts.
8. `owl.replay` and `owl.viz` read recorded evidence without mutating the simulation.

## Repository map

- `src/owl/`: integrated simulation, GPU, recording, counterfactual, CADC, and replay code.
- `src/owl_raqic/`: standalone RAQIC mathematics and validation package.
- `configs/`: local, experimental, GPU, counterfactual, and CADC configurations.
- `scripts/`: current execution, certification, training, evaluation, and packaging tools.
- `tests/`: scientific contracts, recovery checks, parity tests, and failure-path tests.
- `schemas/`: machine-readable simulation and CADC configuration contracts.
- `docs/`: architecture, mathematical contracts, method notes, and references.
- `artifacts/final_certificates/`: compact Phase 2.5 and Phase 3 provenance receipts.

## Local setup and smoke test

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/validate_all_configs.py
python -m pytest -q tests/test_bounds.py tests/test_probability.py \
  tests/test_raqic_dense_equivalence.py
python -c "from owl.core.config import load_config; from owl.engine.loop import run_headless; c=load_config('configs/mvp.yaml'); s,m=run_headless(c,max_steps=5); print(s.tick,len(m))"
```

PowerShell activation uses `.venv\Scripts\Activate.ps1`. GPU, recording, Qiskit, and CADC
workflows use the corresponding optional dependency groups in `pyproject.toml`.

## Main CADC workflow

```bash
python scripts/run_cadc_phase1_acceptance.py --help
python scripts/run_cadc_phase3_acceptance.py --help
python scripts/plan_cadc_phase4_corpus.py --help
python scripts/run_cadc_phase4_corpus.py --help
python scripts/build_cadc_phase4_dataset.py --help
python scripts/train_cadc_phase4.py --help
python scripts/evaluate_cadc_phase4.py --help
python scripts/certify_cadc_phase4.py --help
```

These commands expose required paths and configuration arguments through `--help`.
Target-GPU certification must run on the declared NVIDIA environment; local CPU success is
not a substitute for CUDA, Aer-GPU, NCCL, EGL, memory, or profiling evidence.

## Evidence and limitations

Phase 2.5 and Phase 3 certificates are retained as compact provenance. Large runs, Parquet
tables, checkpoints, model weights, logs, deployment backups, and historical repair bundles
are intentionally excluded. Phase 4 reduced-development results retain their support
limitation: reproduction/topology actions were not executable in the development corpus,
so that family remains out of support and Phase 5 is not unlocked by that evidence.

This project uses simulated data only and does not contain human-subject data. It does not
require CITI or institutional-review-board approval. The code is provided for private course
review and should not be redistributed without the author's permission.
