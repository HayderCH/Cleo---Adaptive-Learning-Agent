# Risks, Metrics, and Success Criteria

## Risks & Controls
- Hallucinated/incorrect items → Quality Gate (NLI, banned claims, duplication).
- Over-adaptation oscillation → Hysteresis + cooldown (3 questions).
- Cognitive overload → Attention index gates high-effort tasks.
- Sparse diagnostics → Synthetic logs + small manual annotation.
- Privacy → Pseudonymized IDs; consent for free-text.

## Core Metrics (Targets)
- Efficiency: −25% questions-to-0.8 mastery vs random baseline.
- Retention: +10–15% 48h delayed test uplift vs static.
- Calibration: −15% Brier after 5 sessions.
- Bloom adherence: < 10% deviation per level.
- Discrimination: Correct − best distractor ≥ 0.35.
- Quality: hallucination/contradiction < 2%; duplication < 5%.

## Reporting
- Weekly: mastery curves, Bloom distribution, calibration trend, quality reports.
- Per ablation: deltas on efficiency, retention, calibration.