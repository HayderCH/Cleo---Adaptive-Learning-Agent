# Next Slice Plan: From COA ->> GEN (Generate Question)

This document outlines what we will build starting at the sequence step:

- COA ->> GEN: Generate question (subject + strategy)

Teammates are handling everything before this (scrape/fine-tune/deploy). We’ll focus on the online loop from question generation to telemetry, scoring, and adaptivity.

---

## 1) Small contract (inputs/outputs)

Inputs to generator (GEN):

- subject: string (e.g., "calculus")
- focus_concepts: string[] (tags aligned to your taxonomy)
- difficulty_target: number (0..1; 0.5 = medium)
- bloom_target: one of [remember, understand, apply, analyze, evaluate, create]
- item_type: one of [mcq, open]
- learner_snapshot: object (minimal LM fields we need; e.g., per-concept mastery 0..1)
- policy_overrides: optional object (e.g., length limits)

Outputs (Question):

- id: string (uuid4)
- stem: string
- choices: string[] (MCQ only; length 3–5) and correct_index: number (MCQ only)
- answer_expl: string (short rationale)
- meta: { concepts: string[], difficulty_est: number, bloom: string, source: string, version: string }

Error modes:

- 400 for invalid inputs; 422 for failed quality gate; 503 for generator backend unavailable

Success criteria:

- > 95% of generated items pass quality gate; MCQ items have exactly one correct option; no PII/toxicity; render in UI.

---

## 2) Minimal pipeline (happy path)

1. COA builds strategy (difficulty/bloom/concepts) from LM and policy
2. GEN.generate(payload) -> Question (stub model first)
3. quality_gate.validate(Question) -> ok or reject with reasons
4. UI shows question; user answers (+ optional self-report)
5. score.answer(question, response) -> score, correctness, latency
6. telemetry.log events (QUESTION_SHOWN, RESPONSE_RAW, RESPONSE_SCORED)
7. COA updates LM, updates policy, loops to next question

---

## 3) Components to implement now

- services/qgen/

  - generator.py: def generate(req: GenerateRequest) -> Question
    - v0: rule-based templates with seeded randomness, no model dependency
    - v1 (later): call HF model or your fine-tuned transformer
  - quality_gate.py: solvability, single-correct (MCQ), ambiguity heuristics, banned words/PII list, length checks
  - schemas.py: Pydantic v2 models (GenerateRequest, Question)
  - README.md: how to run locally

- services/coach/

  - strategy.py: banded difficulty controller with hysteresis; bloom mixer (e.g., [0.5,0.3,0.2])
  - learner_model.py: minimal snapshot (dict persisted to data/processed/lm/{user_id}.json)

- services/ui/

  - app.py (Streamlit): render item, capture answer + optional affect, call telemetry

- scoring/

  - mcq.py: exact match check; compute correctness, latency_ms
  - open.py (later): keyword baseline with thresholds

- scripts/assemble_dataset.py
  - Join raw events -> processed samples aligned to data/schema.json

---

## 4) Event/API contracts we will use

Existing Telemetry service (already implemented):

- POST /events: append JSONL (raw)
- POST /ingest_sample: append JSONL (processed)

Event names & shapes (raw):

- QUESTION_SHOWN: { question_id, subject, concepts[], difficulty_est, bloom }
- RESPONSE_RAW: { question_id, user_answer, affect?: { mood, arousal }, t_start, t_end }
- RESPONSE_SCORED: { question_id, correctness: 0/1, score: number, latency_ms }
- STRATEGY_UPDATE: { difficulty_target, bloom_target, focus_concepts[] }

We’ll keep these aligned with data/schema.json and extend if needed.

---

## 5) Edge cases to handle

- Generator returns empty/duplicate choices (MCQ) -> gate rejects
- Infeasible payload (e.g., bloom=create with trivial concept) -> degrade gracefully to understand/apply
- Timeouts from model backend -> fallback to template generation
- UI reloads mid-item -> idempotent QUESTION_SHOWN logging (same question_id)
- No affect provided -> proceed without it

---

## 6) Tests (minimal, fast)

- generator_test.py: returns well-formed Question; MCQ has one correct_index
- quality_gate_test.py: rejects ambiguous/broken items; accepts clean ones
- scoring_test.py: MCQ scoring correctness; latency computed
- strategy_test.py: band controller raises/lowers difficulty with hysteresis

---

## 7) Milestones (2–4 short sessions)

- M1: Stub generator + gate + unit tests (no UI). CLI script prints one item.
- M2: Wire scoring + telemetry (QUESTION_SHOWN, RESPONSE_RAW, RESPONSE_SCORED). Smoke test end-to-end without UI.
- M3: Minimal Streamlit UI to answer one item; optional self-report affect; verify JSONL writes.
- M4: Strategy controller + LM snapshot persistence; next-item loop.

---

## 8) Try-it flow (after M3)

- Start telemetry service (as in services/telemetry/README.md)
- Run Streamlit UI -> presents item -> answer -> check data/raw/_.jsonl and data/processed/_.jsonl

---

## 9) Out of scope for this slice (later)

- Full transformer integration and prompt engineering
- Advanced gate (NLI entailment, toxicity model, paraphrase checks)
- Free-text semantic grading
- Spaced repetition scheduling beyond simple bands

---

## 10) File map we will add

- services/qgen/generator.py, quality_gate.py, schemas.py, README.md
- services/coach/strategy.py, learner_model.py
- services/ui/app.py (Streamlit)
- scoring/mcq.py
- scripts/assemble_dataset.py
- tests/ (matching files)

If you want, I can scaffold these files now with stubs and unit tests so the team can start filling them in.
