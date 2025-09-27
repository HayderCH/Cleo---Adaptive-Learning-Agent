# Adaptive Learning Companion — Technical Architecture Deck

## Objectives

- Deliver adaptive practice that is relevant, explainable, and stable.
- Optimize for mastery (per concept), calibration, and retention while managing load/affect.

## Architecture at a glance

- Agents: Pedagogical, Memory, Attention, Emotional.
- Shared Learner Model (single source of truth) + Coach (strategy/orchestration).
- Quality/Safety Gate for generated content.
- Unified Telemetry stream for all events.
- Diagrams: see `docs/architecture-option-A.md` and `docs/diagrams/*`.

## Data & Model pipeline (offline)

- Web Scraper → Fine-tuning → Q‑Gen (AI)
  - Scrape subject corpus (web + syllabus); clean/dedupe.
  - Fine‑tune a transformer for domain-appropriate question generation.
  - Deploy the fine‑tuned generator (Q‑Gen) used at runtime.
- Monitoring: domain coverage, hallucination/contradiction rate, question quality audits.

## Runtime interaction loop (online)

- Diagram: `docs/diagrams/interaction-sequence.mmd`
- Steps (collapsed roles):
  1. Adaptivity → Q‑Gen: generate question (subject + strategy).
  2. Q‑Gen → UI: question (stem/options/hints).
  3. UI → Learner: show; Learner answers.
  4. UI → Adaptivity: log response (correctness, time, confidence, optional affect).
  5. Adaptivity: update Learner Model + strategy; loop.

## Telemetry & contracts

- Service: FastAPI (`services/telemetry/main.py`).
- Endpoints:
  - POST `/events` → append EventEnvelope to `data/raw/events_YYYYMMDD.jsonl`.
  - POST `/ingest_sample` → append Sample to `data/processed/samples_YYYYMMDD.jsonl`.
- EventEnvelope (type, source, timestamp, payload), Sample (segment, question, response, affect,…).
- Append‑only JSONL, day‑rotated; schema aligned with `data/schema.json`.

## Adaptivity policy

- Target rolling accuracy band: 70–85% with hysteresis to prevent oscillations.
- Controls: difficulty, Bloom mix, focus concepts, pacing; optional tips.
- Diagram: Coach policy timeline `docs/diagrams/coach-policy-timeline.mmd`.

## Attention & Emotional signals

- Attention: latency_z, error_streak, hint usage, abandonment.
- Emotional: self‑report (valence/arousal) and text‑inferred affect (frustration/demotivation).
- Signals modulate strategy (e.g., ease difficulty, add encouragement, suggest break).

## Memory & Spaced Repetition

- Stability/forgetting model drives next review scheduling.
- Diagram: `docs/diagrams/spaced-review-gantt.mmd`.

## Quality & Safety Gate

- Validates generated questions/answers/feedback: NLI consistency, duplication, tone.
- Reject/fix loop before UI delivery.

## Infra (dev) & runbook

- Python 3.11 venv; GPU available (Torch CUDA 12.1) for future model work.
- Start API (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m uvicorn services.telemetry.main:app --host 127.0.0.1 --port 8000 --reload
```

- Smoke test:

```powershell
$env:PYTHONUNBUFFERED=1; .\.venv\Scripts\python.exe services\telemetry\client_smoke_test.py
```

- Outputs:
  - Raw events: `data/raw/events_YYYYMMDD.jsonl`
  - Samples: `data/processed/samples_YYYYMMDD.jsonl`

## Metrics & evaluation

- Efficiency: questions‑to‑0.8 mastery (≥ −25% vs baseline).
- Retention: 24–48h uplift (+10–15%).
- Calibration: Brier improvement (≥ 15%).
- Bloom adherence: <10% deviation; Quality: contradictions <2%.

## Demo script (2–3 min)

- Show interaction sequence diagram.
- Start API and run smoke test; show appended JSONL files.
- Explain policy band + affect callout; show Gantt for spaced reviews.

## Roadmap

- Minimal UI for responses + self‑report to Telemetry.
- Dataset assembler from events → samples; analytics notebooks.
- Q‑Gen training pipeline automation; Quality Gate integration; online monitoring.

---

Presenter notes

- Emphasize the closed loop: generate → answer → log → update → adapt.
- Offline (scrape → fine‑tune) enables relevance; online telemetry enables adaptivity.
- Safety and stability (Quality Gate + hysteresis) are first‑class.
