# Dataset Card — Adaptive Learning Companion (ALC-EDU-Emo)

## Overview

Production dataset for adaptive learning with AI-powered question generation and emotional support:

- French programming content corpus with RAG retrieval
- AI-generated questions using Phi-3.5 transformer with 8-bit quantization
- Semantic similarity evaluation using sentence-transformers
- Real-time emotion analysis with AI-generated personalized advice
- Telemetry data from user interactions and learning sessions

## Composition

- **Content Corpus**: French programming documentation, chunked and indexed for RAG
- **Questions**: AI-generated using Phi-3.5-mini-instruct with Bloom taxonomy alignment
- **Evaluation**: Semantic similarity scoring (560x faster than basic text matching)
- **Emotional Data**: Real-time affect detection + AI-generated personalized support
- **Telemetry**: User interaction events, response times, confidence levels, learning progress

## Technical Implementation

### Models & Processing

- **Question Generation**: Phi-3.5-mini-instruct (7B) with 8-bit quantization
- **Answer Evaluation**: Sentence-transformers (all-MiniLM-L6-v2) for semantic similarity
- **Emotion Analysis**: emotion-english-distilroberta-base transformer
- **Emotional Advice**: Shared Phi-3.5 pipeline for personalized AI support

### Data Pipeline

- **Content Processing**: French programming documentation → chunking → vector indexing
- **Question Generation**: RAG retrieval + Phi-3.5 generation with quality validation
- **Response Processing**: Semantic similarity scoring against correct answers
- **Emotion Processing**: Text analysis + AI-generated personalized advice
- **Telemetry Collection**: FastAPI service logging all user interactions

## Collection & Generation

- **Content Sources**: Open-license French programming documentation
- **Question Generation**: Template-based + LLM augmentation with quality gates
- **Synthetic Data**: Generated for testing and development scenarios
- **Real Data**: User interaction telemetry with consent and privacy protection

## Quality Assurance

- **Question Validation**: NLI consistency checks, duplication filtering, Bloom alignment
- **Answer Evaluation**: Semantic similarity with fallback to basic matching
- **Emotion Analysis**: Transformer-based classification with confidence scoring
- **Advice Generation**: AI-powered personalized support with safety filtering

## Splits & Usage

- **Training Data**: Synthetic data for model development and testing
- **Real Data**: User interaction logs for system improvement and research
- **Content Corpus**: Static French programming documentation for RAG retrieval
- **Evaluation Data**: Held-out sets for performance validation

## Ethics & Privacy

- **Consent Management**: Explicit user consent for data collection and emotional analysis
- **Data Minimization**: Only necessary data collected for learning optimization
- **PII Protection**: No personally identifiable information stored
- **Emotional Data**: Opt-in only with clear privacy controls
- **AI Safety**: Quality gates and safety filtering for generated content

## Performance Metrics

- **Question Quality**: NLI consistency > 95%, duplication < 2%, Bloom adherence > 90%
- **Answer Evaluation**: Semantic similarity accuracy > 85% vs human judgment
- **Emotion Detection**: Classification accuracy > 80% on validation sets
- **System Performance**: Sub-second response times, 50% VRAM reduction via quantization

## Future Enhancements

- Multi-language content expansion
- Voice-based emotion analysis
- Advanced personalization features
- Integration with additional educational content sources
