# openhup-schemas

The single source of truth for every type that crosses a process boundary in OpenHup.

`backend`, `vision-service`, and `camera-agents` all depend on this package, so an observation
emitted by a detector and consumed by the skill engine is validated against the *same* model on
both sides. The frontend gets TypeScript types generated from the JSON Schema exported here, so a
schema change that breaks the UI breaks it at build time rather than at runtime.

```
openhup_schemas/
├── common.py        durations, ULIDs, slugs, time windows, every shared enum
├── observation.py   Observation + Signal — the vision → bus contract
├── skill.py         Skill, conditions, effects, resolve specs, limits
├── camera.py        Camera + Anchor (the ROI that skills actually watch)
├── task.py          Task, MicroStep, Alert
├── metrics.py       MetricPoint, Goal, WeeklyReport
├── personality.py   personality config + boundary rules
└── events.py        bus envelope and topic names
```

## Design notes

**Durations are written the way humans write them.** `15m`, `1h30m`, `4h`, `2d` all parse to
`timedelta` and serialise back to the compact form, so a skill YAML round-trips unchanged.

**IDs are ULIDs**, generated in-process with no dependency (`new_ulid()`). They sort
lexicographically by creation time, which makes Redis stream IDs, database primary keys, and
episode identifiers all naturally ordered without a separate timestamp index.

**User-facing identifiers are slugs** (`kitchen.counter`, `stove-burner-safety`) so config files
and URLs stay readable. Machine-generated rows use ULIDs.

## Usage

```python
from openhup_schemas import Observation, Skill, load_skill_yaml

skill = load_skill_yaml(open("examples/skills/kitchen-clutter-buster.yaml").read())
assert skill.effect.type == "task"
```

## Generating artifacts

```sh
# JSON Schema for every public model → jsonschema/
uv run python -m openhup_schemas.export jsonschema/

# TypeScript types for the frontend (requires node)
pnpm dlx json-schema-to-typescript -i 'jsonschema/*.json' -o typescript/generated/
```

Both outputs are generated, never committed; CI regenerates and diffs them to catch drift.
