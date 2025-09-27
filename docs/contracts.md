# API Contracts (Events & Schemas)

## Envelope
```json
{
  "id": "uuid",
  "type": "RESPONSE_RAW | DIAGNOSTIC_EVENT | STRATEGY_UPDATE | QUESTION_PLAN | GENERATED_QUESTIONS | REVIEW_TASK | QUALITY_REPORT",
  "source": "Pedagogical|Attention|Emotional|Memory|Coach|QualityGate",
  "timestamp": 1737691221,
  "payload": { }
}
```

## Question
```json
{
  "id": "q_5401",
  "segment_id": "seg_023",
  "concept_focus": ["Adam optimizer","Momentum"],
  "bloom": "Analyze",
  "type": "Compare|Cloze|Apply|Diagnose",
  "stem": "Why might Adam converge faster on sparse gradients?",
  "answer_key": "It adapts per-parameter learning rates using first and second moment estimates.",
  "distractors": [
    "It increases batch size automatically",
    "It guarantees a global minimum",
    "It doubles the learning rate when gradients are sparse"
  ],
  "difficulty_estimate": 0.58,
  "mastery_gain_pred": 0.11,
  "quality_checks": {
    "nli_contradiction": false,
    "duplication": false,
    "tone_ok": true
  }
}
```

## Response Raw
```json
{
  "question_id": "q_5401",
  "user_answer": "It increases batch size automatically",
  "confidence_reported": 85,
  "latency_ms": 8740,
  "correctness": false
}
```

## Diagnostic Event
```json
{
  "question_id": "q_5401",
  "concept_focus": ["Adam optimizer"],
  "error_type": "Concept_Substitution|Recall_Gap|Mechanistic|Overconfidence|Underconfidence|Guessing",
  "severity": 0.7,
  "signals": {
    "latency_z": 1.3,
    "error_streak": 2,
    "hint_usage": 0.0,
    "sentiment": -0.45,
    "frustration_prob": 0.41
  },
  "misconception_cluster_id": "MC_OPT_BATCH_SIZE"
}
```

## Learner Model (shared)
```json
{
  "learner_id": "anon_17",
  "concepts": {
    "Adam optimizer": { "mastery": 0.63, "calibration_error": 0.18, "exposures": 5, "last_review": 1737687000 }
  },
  "bloom_distribution": { "Remember":0.18, "Understand":0.30, "Apply":0.22, "Analyze":0.12, "Evaluate":0.10, "Create":0.08 },
  "pacing": { "target_success_band": [0.7,0.85], "current": 0.64 },
  "affect": { "frustration": 0.22, "fatigue": 0.11 },
  "misconceptions": { "MC_OPT_BATCH_SIZE": { "count": 2, "active": true } }
}
```

## Strategy Update
```json
{
  "action": "ADJUST_PLAN",
  "parameters": {
    "difficulty_delta": -0.05,
    "target_bloom_shift": { "Understand": +0.10, "Analyze": -0.05 },
    "focus_concepts": ["Adam optimizer","Bias correction"],
    "insert_contrast_pairs": [["Adam optimizer","SGD"]],
    "schedule_review": ["Momentum"]
  },
  "tips_for_user": [
    "You often link speed to batch size—review how adaptive moments work.",
    "Try a one-sentence difference: 'Adam differs from Momentum because …'"
  ]
}
```

## Review Task
```json
{
  "concept": "Momentum",
  "due_at": 1737770000,
  "recommended_item": "cloze",
  "stability": 0.42
}
```

## Quality Report
```json
{
  "batch_id": "b_102",
  "hallucination_rate": 0.0,
  "nli_contradictions": 0,
  "duplication_rate": 0.02,
  "readability_grade": 11.5
}
```