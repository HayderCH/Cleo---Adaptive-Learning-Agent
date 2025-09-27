# Adaptive Learning Companion — Option A Architecture

From passive reading to cognitive, adaptive mastery. A multi-agent educational assistant that optimizes mastery, retention, calibration, and emotional well-being.

## What this repo contains

- Architecture and contracts for a 4-agent system + Coach/Orchestrator
- CRISP-DM plan, dataset card, annotation guidelines
- Configs for difficulty/Bloom/spacing and emotion interventions
- Roadmap and evaluation metrics to guide issues/PRs

## Start here

- Architecture: [docs/architecture-option-A.md](docs/architecture-option-A.md)
- Event/JSON Schemas: [docs/contracts.md](docs/contracts.md)
- CRISP-DM plan: [docs/crisp-dm-plan.md](docs/crisp-dm-plan.md)
- Dataset card: [docs/dataset-card.md](docs/dataset-card.md)
- Emotion annotation: [docs/annotation-guidelines-emotion.md](docs/annotation-guidelines-emotion.md)
- Configs: [configs/agents.yaml](configs/agents.yaml), [configs/emotion.yaml](configs/emotion.yaml)
- Data schema: [data/schema.json](data/schema.json)
- Roadmap: [docs/roadmap.md](docs/roadmap.md)
- Risks & Metrics: [docs/risks-metrics.md](docs/risks-metrics.md)

## Ongoing progress

- See the living progress log: [docs/progress.md](docs/progress.md)

## Suggested next steps

1. Open issues from [docs/roadmap.md](docs/roadmap.md) milestones.
2. Scaffold services from [docs/contracts.md](docs/contracts.md).
3. Begin dataset bootstrapping per [docs/dataset-card.md](docs/dataset-card.md) using [data/schema.json](data/schema.json).

## Environment

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ethics

Pseudonymized IDs; explicit consent for free-text; supportive tone; no diagnostic claims.
