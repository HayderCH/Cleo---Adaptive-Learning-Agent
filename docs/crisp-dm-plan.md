# Adaptive Learning Companion — CRISP-DM Plan

## 1) Business Understanding
- Goal: Improve learning efficiency, retention, and metacognitive calibration while maintaining motivation and emotional well‑being.
- Success: Efficiency −25% questions-to-0.8; Retention +10–15% (48h); Calibration −15% Brier; Emotion −30% frustration; −20% drop-off; +0.5 motivation (1–5).

## 2) Data Understanding
- Sources: ML/DS segments; generated Qs; interaction logs; emotional signals (text/self-report; optional voice later).
- Risks: Imbalanced affect labels; cold start; domain sentiment.

## 3) Data Preparation
- Build custom dataset: segments ↔ questions ↔ responses; emotion labels; derived signals.
- QA: dedupe questions; NLI contradiction checks; balance concepts/affect.

## 4) Modeling
- Cognitive: mastery update (Bayesian); spacing (SM-2 variant).
- Emotional: text affect classifier (GoEmotions fine-tune) + interaction-signal model; late fusion.
- Quality Gate: NLI contradiction; tone/duplication filters.

## 5) Evaluation
- A/B: Adaptive+Emotion vs Adaptive-neutral (n=5–10).
- Simulation: learner archetypes (Overconfident, Slow Retainer, Easily Frustrated).
- Ablations: remove Memory/Attention/Emotional/Quality Gate.

## 6) Deployment
- MVP UI (Streamlit/Gradio); dashboards; ethics (pseudonymization, consent).