# Dataset Card — Adaptive Learning Companion (ALC-EDU-Emo)

## Overview
Custom dataset for ML/DS topics with cognitive and emotional annotations:
- Segments (technical text), questions (Bloom), responses (correctness, latency, confidence), emotional signals (labels + valence/arousal), derived interaction features.

## Composition
- 200–400 segments across 2–3 topics; 3–6 Qs/segment; balanced Bloom.
- Responses: synthetic (pilot) + real (small study).
- Affect labels: Frustration, Demotivation, Stress, Neutral (+ valence, arousal).

## Collection
- Open-license content; questions via templates + LLM; Quality Gate verification.
- Emotions via self-report micro-prompts + text sentiment + interaction proxies.

## Annotation
- Two-pass: heuristic/prompt → human correction; κ ≥ 0.6 target.

## Splits
- Train/Val/Test by segment; user-level split for logs.

## Ethics
- Consent; opt-out for affect; PII avoided; free-text stored only with consent.