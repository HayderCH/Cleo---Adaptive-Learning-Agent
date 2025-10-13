# Adaptive Learning Companion — Slide Deck

---

# Slide 1 — Title

Adaptive Learning Companion
AI-powered personalized education with emotion-aware tutoring and RAG-grounded question generation

Presenter: HayderCH
October 2025

Notes: One-sentence elevator pitch, 30s.

---

# Slide 2 — Problem

- One-size-fits-all education fails many learners
- Lack of personalization in assessment and pacing
- No emotional awareness or adaptive support in typical e-learning
- Teachers need transparency into what the AI uses to grade and create questions

Notes: Give an example student struggle (20s)

---

# Slide 3 — Solution Overview

Adaptive Learning Companion: a multi-agent AI system that:

- Generates context-grounded questions (RAG + Phi-3.5)
- Scores answers semantically (sentence-transformers)
- Tracks learner mastery and schedules reviews (spaced repetition)
- Detects affect and supplies personalized interventions
- Offers full transparency via RAG visualization in the UI

Notes: 40s

---

# Slide 4 — System Architecture (Text Only)

Components (text-only, diagrams will be added later):

- UI: Streamlit app with RAG visualization and emotion widgets
- Telemetry: FastAPI service collecting metrics and events
- Q-Gen: Transformer backend (Phi-3.5 local, 8-bit) + template fallback
- Memory: Spaced repetition scheduler and learner model
- Attention: Interaction trace processing and load detection
- Emotion: Affect model and advice generator
- Coach: Orchestrator that selects strategy and policies
- Data: JSONL corpus, scrapers, and synthetic data

Notes: 60s — mention where diagrams will be placed

---

# Slide 4.1 — CRISP‑DM & Data Pipeline (Scraping-first)

We followed a CRISP‑DM process for data collection and preparation; key points below.

1. Business Understanding

- Objective: Build a domain-specific corpus (French programming & curricula) to ground question generation and evaluation.

2. Data Understanding (Scraping)

- Sources: official course pages, documentation, tutorials, code examples, syllabus PDFs
- Tooling: custom web scraper (`scripts/serp_scraper.py`) + manual curation
- Outputs: raw HTML/text stored into `data/raw/` and processed JSONL in `data/processed/`

3. Data Preparation

- Cleaning: strip boilerplate, normalize whitespace, remove navigation text
- Chunking: create semantically-coherent text chunks for retrieval (stored as `chunks_*.jsonl`)
- Deduplication & QA: NLI checks, duplicate removal, metadata enrichment (source, url, date)
- Indexing: build retrieval index (embedding-based or simple keyword index)

4. Modeling & Fine-tuning

- Fine-tune or adapt LLM prompts using curated examples (optional)
- Build embedding index with sentence-transformers for semantic retrieval

5. Evaluation & Deployment

- Offline evaluation: held-out QA, A/B tests, simulation of learner archetypes
- Deployment: serve processed JSONL + index to RAG pipeline

Reference: `docs/crisp-dm-plan.md` for full process and evaluation plan

Notes: 60s — emphasize scraping provenance and QA

---

# Slide 5 — RAG Pipeline (Visual)

```mermaid
flowchart TB
  subgraph Input[Input Request]
    A[Subject & Focus Concepts]
    B[Constraints: Bloom, Difficulty]
  end

  subgraph Retrieval[Retrieval]
    C[Search Corpus]
    D[Score & Rank Chunks]
    E[Top-K Chunks]
  end

  subgraph Augment[Augmentation]
    F[Compose Prompt with Context]
  end

  subgraph Generation[LLM Generation]
    G[Phi-3.5 (local)]
    H[Generate JSON-formatted Question]
  end

  subgraph Post[Post-processing]
    I[Parse JSON]
    J[Quality & Safety Gate]
    K[Return Question]
  end

  A --> C
  B --> F
  C --> D --> E --> F
  F --> G --> H --> I --> J --> K

  style Input fill:#f9f,stroke:#333,stroke-width:1px
  style Retrieval fill:#ffb,stroke:#333,stroke-width:1px
  style Augment fill:#bbf,stroke:#333,stroke-width:1px
  style Generation fill:#bfb,stroke:#333,stroke-width:1px
  style Post fill:#fdd,stroke:#333,stroke-width:1px
```

Notes: 40s

---

# Slide 6 — Key Features

- Multi-format questions (MCQ / open / code)
- Semantic evaluation (fast, robust)
- Emotion-aware pacing & interventions
- Spaced repetition and mastery tracking
- RAG transparency for educators

Notes: 30s

---

# Slide 7 — Demo Plan

1. Start backend: `python services/telemetry/main.py`
2. Start UI: `streamlit run services/ui/app.py --server.port 8502`
3. Generate a question with transformer backend
4. Open RAG visualization → inspect retrieved chunks
5. Submit answer and watch telemetry updates

Notes: 60s demonstration flow

---

# Slide 8 — Performance & Engineering

- Phi-3.5 local on RTX 4060 (CUDA 12.6)
- 8-bit quantization for VRAM savings
- 560x speedup in evaluation using sentence-transformers
- Monitoring: generation latency, accuracy, and alerts

Notes: 30s

---

# Slide 9 — Roadmap (Phase 2 focus)

- Code Execution Validation (sandboxed) — high priority
- Multi-language expansion beyond French
- Advanced emotional interventions
- Rich progress dashboards for educators

Notes: 30s

---

# Slide 10 — Security & Ethics

- Sandboxed execution for code answers
- Quality gate to reject unsafe or off-topic content
- Telemetry privacy: anonymized user IDs and opt-in
- Transparent RAG sources to avoid hallucination

Notes: 30s

---

# Slide 11 — How to Run & Contribute

- Clone repo, create venv, install requirements
- Start telemetry and UI (see demo slide)
- Tests: `pytest` (project has coverage and CI checks)
- Contact & repo: https://github.com/HayderCH/Cleo---Adaptive-Learning-Agent

Notes: 20s

---

# Slide 12 — Acknowledgements & Next Steps

- Research foundation: spaced repetition, affective computing
- Engineering: model quantization, shared model instances
- Next steps: implement code execution validation

Notes: closing, 30s
