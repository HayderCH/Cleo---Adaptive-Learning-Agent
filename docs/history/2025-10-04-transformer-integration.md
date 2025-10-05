# 2025-10-04 – Transformer Integration & Hardening

## Summary

- Installed CUDA 12.1 nightly build of PyTorch 2.6.0.dev to satisfy the new `torch.load` safety requirement enforced by 🤗 Transformers.
- Converted the emotion classifier weights to `model.safetensors`, allowing the emotion pipeline to run offline without triggering the CVE block.
- Downloaded the Phi-3.5 mini question-generation model (two shard safetensors) and verified GPU loading via `device_map="auto"` with offloading for constrained VRAM.
- Hardened `services/qgen/backends/transformer.py`:
  - Added dtype compatibility shim and fallbacks for legacy `torch_dtype` configs.
  - Implemented resilient JSON parsing/repair plus normalization for choices, answer keys, and explanations (handles dict/list variants and infers missing indices).
  - Enriched prompts with MCQ/open-specific guidance to meet quality gate expectations (canonical answers, stem length, max 4 choices).
  - Added inference heuristics so missing `correct_index`/`answer_key` fields no longer trigger template fallback.
  - Introduced automatic retry that reinitializes the pipeline without `device_map` when Accelerate leaves layers on the meta device, ensuring local generation proceeds instead of falling back to templates.
  - Added validation/retry for open-response outputs so malformed generations re-run instead of falling back to MCQ templates, logged raw candidates for easier debugging, and normalize stray MCQ-style fields back into canonical open-response form.
- Updated `services/qgen/generator.py` to automatically retry transformer backend loading if a prior failure cached the template fallback, so Streamlit recovers without manual restarts.
- Updated `configs/qgen.yaml` to use `dtype: float16` for the local Phi deployment.
- Re-ran `pytest tests/test_qgen.py` and manual generation sanity checks to confirm transformer backend remains stable post-changes.

  ## 2025-10-05 – Backend Source Attribution Fix

  - Normalized transformer-generated questions to report `meta.source="transformer"` while preserving the model-supplied provenance in a new `meta.origin` field. This prevents the Streamlit UI from flagging legitimate transformer output as a template fallback while retaining the original citation metadata.
  - Re-ran `pytest tests/test_qgen.py` to verify schema changes and backend adjustments.
  - Added an additional open-response normalization pass that strips any stray MCQ fields before validation and derives a canonical answer if the model omits one, eliminating unnecessary template fallbacks.
  - Revalidated with `pytest tests/test_qgen.py` after the normalization fix.
  - Replaced the heuristic difflib scorer with a transformer-backed evaluator that grades open responses via JSON scoring/rationale, falling back to the old similarity check if the model is unavailable. Updated the Streamlit UI to surface the new score and rationale.
  - Ran the full pytest suite to confirm the scorer integration behaves across services.
  - Addressed quality-gate “answer key missing” by clearing any residual MCQ fields when the transformer returns an open response, ensuring the gate evaluates the item with a proper canonical answer.
  - Re-ran the full pytest suite after the normalization fix (14 passed, 3 skipped).
  - Relaxed the quality gate’s answer-length requirement so short numeric responses no longer trigger false “answer key missing” failures while still catching truly empty answers. Full pytest suite remains green (14 passed, 3 skipped).

## Follow-ups

- Monitor Streamlit UI runs for rare transformer outputs that still fail structural validation; capture raw JSON and extend repair heuristics if needed.
- Keep an eye on upstream PyTorch nightly warning (`torch.classes … __path__._path`) for resolution or downgrade once 2.6 stable is available.
