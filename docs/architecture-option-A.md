# Adaptive Learning Companion — Architecture (Option A)

## Goals

- Preserve 4 specialized agents (Pedagogical, Memory, Attention, Emotional).
- Add a shared Learner Model (single source of truth), a Coach/Orchestrator (strategy), and a Quality/Safety Gate (content validation).
- Ensure explainability, stable adaptivity, and measurable impact.

## Component Overview

- Pedagogical Agent: segments content, generates questions, scores answers, renders augmentation packs.
- Memory Agent: models retention; schedules spaced reviews; predicts decay.
- Attention Agent: estimates cognitive load (latency z-score, error streaks, hint usage, abandonment).
- Emotional Agent: infers affect from text (and later voice): frustration, demotivation, stress.
- Coach/Orchestrator: consumes diagnostics + Learner Model; sets difficulty, Bloom mix, focus concepts, spacing; sends tips.
- Learner Model (shared): mastery per concept, calibration error, fatigue/affect estimates, spacing schedule, Bloom coverage stats.
- Quality/Safety Gate (optional): validates generated questions, answers, feedback (NLI consistency, duplication, tone).

## High-Level Dataflow

1. Pedagogical → delivers Segment + Questions (validated by Quality Gate).
2. Learner responds → Response event.
3. Attention + Emotional produce signals; Pedagogical provides correctness; diagnostics assembled.
4. Coach updates Learner Model and sends Strategy Update to Pedagogical/Memory.
5. Memory triggers scheduled reviews; loop continues.

```mermaid
flowchart LR
  SRC[Course Content] --> PED(Pedagogical)
  PED -->|Questions| QG[Quality Gate]
  QG --> UI[UI/Delivery]
  UI -->|Response| LOG[Telemetry]
  LOG --> ATT(Attention)
  LOG --> EMO(Emotional)
  LOG --> PED
  ATT --> COA(Coach/Orchestrator)
  EMO --> COA
  PED --> COA
  MEM(Memory) --> COA
  COA --> LM[(Learner Model)]
  LM --> COA
  COA -->|Strategy Update| PED
  COA --> MEM
```

## Signals

- Attention: latency_z, error_streak, hint_usage_rate, abandonment_flag.
- Emotional: sentiment_score, frustration_prob, demotivation_prob.
- Pedagogical: correctness, discrimination index, Bloom coverage drift, redundancy.
- Memory: stability score, next_review_time.

## Policies

- Target success band: 70–85% rolling accuracy.
- Bloom target (adjustable): R 0.2, U 0.25, A 0.2, An 0.15, E 0.1, C 0.1.
- Hysteresis to prevent oscillations; quality gate for safety.

## Evaluation & Ablations

- Efficiency: Questions-to-0.8 mastery (≥ −25% vs static/random).
- Retention: 48h uplift (+10–15%).
- Calibration: Brier improvement (≥ 15%).
- Bloom adherence: < 10% deviation per level.
- Quality: hallucination/contradiction < 2%.

## Interaction sequence (end-to-end)

```mermaid
sequenceDiagram
  autonumber
  participant SRC as Content Source
  participant PED as Pedagogical
  participant QG as Quality Gate
  participant UI as UI/Delivery
  participant LOG as Telemetry
  participant ATT as Attention
  participant EMO as Emotional
  participant COA as Coach
  participant LM as Learner Model
  participant MEM as Memory
  participant LRN as Learner

  SRC->>PED: Provide content
  PED->>QG: Segment + candidate questions
  QG-->>PED: Validate/fix or reject
  QG-->>UI: Approved questions + packs
  UI->>LRN: Deliver item
  LRN->>UI: Answer
  UI->>LOG: POST /events (RESPONSE_RAW)
  PED-->>LOG: DIAGNOSTIC_EVENT (correctness, item stats)
  LOG-->>ATT: Event stream
  LOG-->>EMO: Event stream
  ATT->>COA: load signals (latency_z, error_streak...)
  EMO->>COA: affect signals (frustration_prob...)
  PED->>COA: item diagnostics
  COA->>LM: Update mastery/calibration/fatigue
  COA->>PED: STRATEGY_UPDATE (difficulty, Bloom mix, focus concepts)
  COA->>MEM: STRATEGY_UPDATE (review budget/policy)
  MEM->>COA: next review recommendations
  COA-->>UI: Tips/encouragement (optional)
  loop Next item
    PED->>QG: Next item proposal
    QG-->>UI: Approved item
    UI->>LRN: Deliver
  end
```

## Component map (non-boxy)

```mermaid
mindmap
  root((Adaptive Learning Companion))
    Delivery
      UI/Delivery
    Shared
      Learner Model
      Quality/Safety Gate
      Telemetry Service
    Agents
      Pedagogical
      Memory
      Attention
      Emotional
      Coach/Orchestrator
    Content
      Source
```

Optional flow alternative (if your Mermaid renderer supports it):

```mermaid
sankey-beta
  SRC,Content  PED,Pedagogical  1
  PED,Pedagogical  QG,Quality Gate  1
  QG,Quality Gate  UI,UI  1
  UI,UI  LOG,Telemetry  1
  LOG,Telemetry  ATT,Attention  0.6
  LOG,Telemetry  EMO,Emotional  0.6
  LOG,Telemetry  PED,Pedagogical  0.3
  ATT,Attention  COA,Coach  1
  EMO,Emotional  COA,Coach  1
  PED,Pedagogical  COA,Coach  1
  COA,Coach  LM,Learner Model  1
  COA,Coach  PED,Pedagogical  1
  COA,Coach  MEM,Memory  1
```

## Telemetry schema view (mindmap)

```mermaid
mindmap
  root((Sample))
    segment
      segment_id
      topic
      text
      concepts[]
    question
      question_id
      bloom
      type
      stem
      answer_key
      distractors[]
    response
      user_id
      correct
      user_answer
      confidence (0-100)
      latency_ms
      hint_used
    affect
      labels
      valence (-1..1)
      arousal (0..1)
      source
    derived_signals
      latency_z
      error_streak
      novelty_score
    intervention
      recommended
      accepted
    consent
      store_free_text
      store_audio
    timestamps
      created_at (iso8601)
      updated_at (iso8601)
```

## Coach policy guidance (timeline + journey)

```mermaid
timeline
  title Coach policy bands
  section Rolling accuracy
    Below 0.70 : decrease difficulty, add scaffolding, keep Bloom level
    0.70–0.85 : maintain strategy; micro-adjust via ATT/EMO load
    Above 0.85 : increase difficulty, diversify Bloom, reduce scaffolding
```

```mermaid
journey
  title Strategy update loop (per batch)
  section Measure
    Collect ATT/EMO/PED signals: 3: Coach
  section Decide
    Compare to target band with hysteresis: 3: Coach
    Choose adjustments (difficulty, Bloom mix, focus concepts): 4: Coach
  section Apply
    Emit STRATEGY_UPDATE to PED/MEM: 3: Coach
    Update LM with deltas: 2: Coach
```

## Spaced review schedule (example)

```mermaid
gantt
  dateFormat  YYYY-MM-DD
  title Spaced Repetition Plan (per concept)
  section Concept A
  Initial Study     :a1, 2025-09-27, 1d
  Review 1 (1d)     :after a1, 1d
  Review 2 (3d)     :3d
  Review 3 (7d)     :7d
  section Concept B
  Initial Study     :b1, 2025-09-27, 1d
  Review 1 (2d)     :after b1, 2d
  Review 2 (5d)     :5d
```
