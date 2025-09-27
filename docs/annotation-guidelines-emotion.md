# Emotion Annotation Guidelines (ALC-EDU-Emo)

## Label Set
- Frustration: annoyance/irritation at task difficulty or repeated failure.
- Demotivation: low willingness to continue; “je n’y arriverai jamais”.
- Stress: pressure/overload; anxious urgency.
- Neutral: none of the above.

Optional continuous: Valence ∈ [-1,1], Arousal ∈ [0,1].

## Sources
- Self-report micro-prompt (1–5).
- Free-text reflections (short).
- Interaction proxies: latency z, error streaks, abandoning items, hint usage.

## Rules
- Prefer self-report over model inference when present.
- Frustration vs Stress: irritation vs pressure.
- Demotivation = low-efficacy statements.
- Multi-label allowed (e.g., Frustration + Stress).

## Quality
- Double-annotate 10% sample; κ ≥ 0.6; weekly adjudication.
- Privacy: pseudonymize; store free-text only with consent.