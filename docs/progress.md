# Project Progress Log

A living, high-level log of what we’ve done and what’s next. Update this file as we move forward.

## Status — 2025-09-26

### Done

- Organized repository structure: moved docs, configs, and data schema to their target folders.
- Added synthetic data generator: `scripts/generate_synthetic_data.py`.
  - Produces `data/processed/samples_*.jsonl` matching `data/schema.json`.
  - Emits example raw `RESPONSE_RAW` events to `data/raw/events_*.jsonl`.
- Set Python to 3.11 and resolved dependency pins (notably numpy==1.26.4).
- Installed GPU-accelerated PyTorch (CUDA 12.1) for RTX 4060:
  - torch==2.4.1, torchvision==0.19.1, torchaudio==2.4.1 with cu121 wheels.

### Decisions

- Data collection will be no-audio for the prototype.
- Affect signals will use BOTH:
  - Simple self-report (1–5 frustration; optional valence/arousal), and
  - Text-only sentiment/affect where available (e.g., short free text).
- One unified event stream (per `docs/contracts.md`) feeds all agents; no per-agent datasets needed.

### Next Up

- Telemetry service (FastAPI) `POST /events` writing append-only JSONL to `data/raw/`.
- Assembler script to convert `data/raw/*.jsonl` into `data/processed/*.jsonl` aligning to `data/schema.json`.
- Minimal Streamlit page with response form + confidence + self-report sliders; posts to telemetry.

### Quick refs

- Generate synthetic data (example):
  ```powershell
  cd "c:\Users\GIGABYTE\projects\Adaptive Learning Companion"
  .\.venv\Scripts\python.exe scripts\generate_synthetic_data.py --users 3 --segments-per-topic 4 --questions-per-segment 3
  ```
- Verify GPU in PyTorch (optional):
  ```powershell
  .\.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  ```
