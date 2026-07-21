# Observer-Window Life Architecture

Observer-Window Life is organized around one canonical runtime object:
`owl.core.state.WorldState`. Computation is structure-of-arrays: dense NumPy
arrays for cell fields, action fields, communication channels, and patch/global
summaries. Sparse Python records are reserved for rare events such as collision,
ingestion, reproduction, merge, split, expulsion, and release.

## Layers

### 1. Physical/classical layer

Implemented by:
- `owl.engine.environment`
- `owl.engine.feeding`
- `owl.engine.health`
- `owl.engine.movement`
- `owl.engine.collision`
- `owl.engine.death`
- `owl.engine.reproduction`
- `owl.engine.topology`

This layer updates food, toxin, signal fields, resource, health, boundary,
movement/collision, reproduction, death, and topology hooks.

### 2. Possibility/actualization layer

Implemented by:
- `owl.engine.utility`
- `owl.engine.authority`
- `owl.engine.actualization`

This layer computes drives, utility scores, feasibility/authority masks, action
logits, normalized possibility distributions, and actualized readouts. Dead or
obstacle cells are forced to REST.

### 3. Fractal/mosaic layer

Implemented by:
- `owl.engine.phase`
- `owl.engine.integration`
- `owl.engine.aggregation`
- `owl.engine.topdown`

This layer computes phase, local synchrony, same-scale coherence, cross-scale
coupling, integration, patch/global summaries, and bounded parent/apex bias.

### 4. Universal communication substrate

Implemented by:
- `owl.engine.communication`
- `owl.engine.sensing`

All OWs can emit and receive signals. Signal style is encoded by mutable
continuous traits, not a dedicated signaler species.

### 5. Main loop

Implemented by:
- `owl.engine.scheduler`
- `owl.engine.loop`

`owl.engine.loop.step` is the only engine function that composes all lower-level
engine systems. Lower-level engine modules must not import `owl.engine.loop`.

### 6. Recording, visualization, experiments, and analysis

Implemented by:
- `owl.record.*`
- `owl.viz.*`
- `owl.experiments.*`
- `owl.analysis.*`

These packages read runtime state or saved outputs. They must not own simulation
rules or mutate the hot-loop state except through explicit viewer-local controls.

## MVP tick order

1. Environment and passive signal fields.
2. Signal reception.
3. Patch/global context and top-down bias.
4. Phase/synchrony/coherence/cross-scale diagnostics.
5. Utility, authority, logits, possibility, and readout actualization.
6. Movement, collision, inhibition.
7. Feeding, repair, signaling, reproduction, topology hooks.
8. Metabolism, memory, signal memory.
9. Conflict, integration, channel trust.
10. Death/release and final patch/global refresh.

## Invariants

Completed ticks should satisfy:
- finite dense fields;
- bounded survival and communication fields;
- normalized possibility vectors for living cells;
- REST readouts for dead/obstacle cells;
- matching cell/action/channel/patch shapes;
- no presentation/recording/analysis imports inside engine modules.
