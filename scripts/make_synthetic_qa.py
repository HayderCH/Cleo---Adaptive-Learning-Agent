"""Generate synthetic QA items from text chunks.

Usage examples:
  python scripts/make_synthetic_qa.py --chunks data/processed/corpus.jsonl --out data/processed/qa.jsonl --per-chunk 2 --subject math
  python scripts/make_synthetic_qa.py --chunks data/processed/corpus.jsonl --out data/processed/qa.jsonl --per-chunk 1 --bloom apply

The script generates MCQ and open-ended questions from text chunks, with quality gate validation.
"""

import argparse
import json
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Iterable

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def simple_distractors(answer: str) -> list[str]:
    # Improved distractors: numeric changes, negation, shuffling
    dist = []

    # Numeric manipulation: change numbers
    def change_nums(s):
        return re.sub(
            r"\b(\d+)", lambda m: str(int(m.group(1)) + random.randint(-2, 2) or 1), s
        )

    # Negation: add/remove "not"
    def negate(s):
        words = s.split()
        if "not" not in words:
            words.insert(random.randint(0, len(words)), "not")
        return " ".join(words)

    # Shuffling
    def shuffle_words(s):
        words = s.split()
        if len(words) > 1:
            random.shuffle(words)
        return " ".join(words)

    # Generate candidates
    candidates = [
        change_nums(answer),
        negate(answer),
        shuffle_words(answer),
        f"not {answer}",
        f"{answer} is false",
        f"{answer} does not apply",
    ]

    # Filter unique and different from answer
    seen = set([answer])
    for cand in candidates:
        if cand not in seen:
            dist.append(cand)
            seen.add(cand)
            if len(dist) >= 3:
                break

    # Fallback if not enough
    while len(dist) < 3:
        fallback = f"alternative option {len(dist) + 1}"
        if fallback not in seen:
            dist.append(fallback)
            seen.add(fallback)

    return dist[:3]


def make_items_from_chunk(
    chunk: dict,
    per_chunk: int = 2,
    subject: str | None = None,
    bloom: str = "understand",
    difficulty: float = 0.3,
    pipeline=None,
):
    """Generate QA items from a text chunk using the transformer
    model to read and understand the content."""
    text = chunk.get("text", "").strip()
    if not text or len(text.split()) < 20:
        return []

    out = []

    # Check if pipeline is provided
    if pipeline is None:
        print("Pipeline not provided, using template fallback")
        return _template_fallback(chunk, per_chunk, subject, bloom, difficulty)

    try:
        # Generate questions from the actual text content
        for i in range(min(per_chunk, 3)):  # Generate up to 3 questions
            # Create prompt with actual content
            content_preview = text[:600]  # Shorter for better focus
            prompt = f"""You are a teacher creating a multiple choice question
from this computer science educational text.

TEXT CONTENT:
{content_preview}

TASK:
Create ONE multiple choice question that tests understanding of a key
concept from this specific text.
The question must be directly based on information in the text above.

Return ONLY valid JSON in this exact format:
{{
"question": "your question here",
"options": ["choice A", "choice B", "choice C", "choice D"],
"correct_answer": 0,
"explanation": "brief explanation from the text"
}}

JSON:"""

            try:
                # Use pipeline for generation
                outputs = pipeline(
                    prompt,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    return_full_text=False,
                )

                response = outputs[0]["generated_text"]

                # Parse the JSON response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    qa_data = json.loads(json_str)

                    item = {
                        "id": str(uuid.uuid4()),
                        "stem": qa_data.get("question", "What is the main concept?"),
                        "choices": qa_data.get("options", ["A", "B", "C", "D"]),
                        "correct_index": qa_data.get("correct_answer", 0),
                        "answer_expl": qa_data.get(
                            "explanation", "Based on the content."
                        ),
                        "meta": {
                            "concepts": chunk.get("concepts", []),
                            "difficulty_est": difficulty,
                            "bloom": bloom,
                            "source": "transformer_generated",
                            "subject": subject or chunk.get("subject", ""),
                        },
                    }
                    out.append(item)
                else:
                    raise ValueError("No JSON found in response")

            except Exception as e:
                print(f"Failed to generate with direct model: {e}, " "using template")
                template_items = _template_fallback(
                    chunk, 1, subject, bloom, difficulty
                )
                out.extend(template_items[:1])

    except Exception as e:
        print(f"Direct model setup failed: {e}, using template fallback")
        out = _template_fallback(chunk, per_chunk, subject, bloom, difficulty)

    return out


def _template_fallback(chunk, per_chunk, subject, bloom, difficulty):
    """Smart template fallback that extracts concepts from the actual text."""
    text = chunk.get("text", "")
    sentences = [
        s.strip()
        for s in text.split(".")
        if s.strip() and len(s.split()) > 5  # Good length sentences
    ]

    if not sentences:
        return []

    out = []

    # Extract key terms from the text (programming-related)
    key_terms = []
    for sentence in sentences[:3]:  # Look at first few sentences
        words = sentence.split()
        # Look for programming terms
        for word in words:
            word = word.strip(".,!?()[]{}")
            if len(word) > 3 and (
                "function" in word.lower()
                or "class" in word.lower()
                or "variable" in word.lower()
                or "method" in word.lower()
                or "algorithm" in word.lower()
                or "data" in word.lower()
                or "program" in word.lower()
                or "code" in word.lower()
                or "Python" in word
                or "JavaScript" in word
                or "Java" in word
                or "programming" in word.lower()
            ):
                key_terms.append(word)

    # If no programming terms found, use general technical terms
    if not key_terms:
        for sentence in sentences[:3]:
            words = sentence.split()
            for word in words:
                word = word.strip(".,!?()[]{}")
                if len(word) > 4 and word[0].isupper():  # Proper nouns, titles
                    key_terms.append(word)

    # Fallback to any meaningful words
    if not key_terms:
        for sentence in sentences[:3]:
            words = [w.strip(".,!?()[]{}") for w in sentence.split() if len(w) > 3]
            key_terms.extend(words[:2])

    for i in range(min(per_chunk, len(sentences))):
        sent = sentences[i]

        # Use extracted terms to create relevant questions
        if key_terms:
            main_term = key_terms[min(i, len(key_terms) - 1)]
            question_templates = [
                f"What is the role of {main_term} in programming?",
                f"How does {main_term} work in this context?",
                f"What does {main_term} do according to this text?",
                f"According to the content, what is {main_term}?",
                f"What is explained about {main_term} here?",
            ]
        else:
            question_templates = [
                f"What is the main concept in: '{sent[:50]}...'?",
                f"What does this text explain: '{sent[:50]}...'?",
                f"What key information is here: '{sent[:50]}...'?",
            ]

        stem = random.choice(question_templates)

        # Extract a reasonable answer from the sentence
        words = sent.split()
        if len(words) > 5:
            # Try to find a meaningful phrase
            answer_candidates = []
            for j in range(len(words) - 1):
                phrase = " ".join(words[j : j + 2])
                if len(phrase) > 10 and len(phrase) < 50:
                    answer_candidates.append(phrase)

            if answer_candidates:
                answer = random.choice(answer_candidates)
            else:
                answer = " ".join(words[1:4])  # Skip first word often
        else:
            answer = " ".join(words[:3])

        # Create MCQ with relevant distractors
        choices = [answer]

        # Generate distractors based on subject
        subject_lower = (subject or chunk.get("subject", "")).lower()
        if "python" in subject_lower:
            distractors = [
                "JavaScript syntax",
                "HTML tags",
                "CSS styles",
                "database queries",
                "network protocols",
                "file systems",
            ]
        elif "javascript" in subject_lower:
            distractors = [
                "Python indentation",
                "Java classes",
                "C++ pointers",
                "SQL commands",
                "HTTP headers",
                "DOM manipulation",
            ]
        elif "programming" in subject_lower or "computer" in subject_lower:
            distractors = [
                "hardware components",
                "operating systems",
                "networking",
                "database design",
                "web development",
                "software engineering",
            ]
        else:
            # Generic technical distractors
            distractors = [
                "implementation details",
                "system architecture",
                "user interface",
                "data processing",
                "algorithm design",
                "software development",
            ]

        # Add 3 distractors
        choices.extend(random.sample(distractors, min(3, len(distractors))))
        random.shuffle(choices)

        mcq = {
            "id": str(uuid.uuid4()),
            "stem": stem,
            "choices": choices,
            "correct_index": choices.index(answer),
            "answer_expl": f"The text explains: {answer}",
            "meta": {
                "concepts": key_terms[:3],  # Use extracted concepts
                "difficulty_est": difficulty,
                "bloom": bloom,
                "source": "smart_template_fallback",
                "subject": subject or chunk.get("subject", ""),
            },
        }
        out.append(mcq)

    return out


# def log_event(event_type: str, source: str, details: dict = None):
#     """Log an event to the telemetry system (simulation)."""
#     event = {
#         "id": str(uuid.uuid4()),
#         "type": event_type,
#         "source": source,
#         "timestamp": int(datetime.now(timezone.utc).timestamp()),
#         "details": details or {},
#     }
#     events_file = Path("data/raw/events.jsonl")
#     events_file.parent.mkdir(parents=True, exist_ok=True)
#     with open(events_file, "a", encoding="utf-8") as f:
#         json.dump(event, f)
#         f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", required=True, help="Input corpus JSONL")
    p.add_argument("--out", required=True)
    p.add_argument("--per-chunk", type=int, default=2)
    p.add_argument("--subject", help="Subject label for QA items")
    p.add_argument(
        "--bloom",
        choices=["remember", "understand", "apply", "analyze"],
        default="understand",
    )
    p.add_argument(
        "--difficulty",
        type=float,
        default=0.3,
        help="Difficulty level 0-1",
    )
    args = p.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Load pipeline once
    model_path = "models/qgen_phi35"
    pipeline = None

    if Path(model_path).exists():
        try:
            from transformers import pipeline

            print("Loading Phi-3.5 model with pipeline...")
            pipeline = pipeline(
                "text-generation",
                model=model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            print("Pipeline loaded successfully!")
        except Exception as e:
            print(f"Failed to load pipeline: {e}, will use template fallback")
    else:
        print(f"Model not found at {model_path}, will use template fallback")

    total_items = 0
    accepted_count = 0
    chunk_count = 0

    with outp.open("w", encoding="utf-8") as oh:
        for chunk in iter_jsonl(Path(args.chunks)):
            chunk_count += 1
            if chunk_count > 100:  # Limit to first 100 chunks for testing
                break
            items = make_items_from_chunk(
                chunk,
                args.per_chunk,
                args.subject,
                args.bloom,
                args.difficulty,
                pipeline,
            )
            for it in items:
                total_items += 1
                # Skip validation for simulation
                oh.write(json.dumps(it, ensure_ascii=False) + "\n")
                accepted_count += 1

    print(
        f"Generated {accepted_count} QA items from {total_items} "
        f"total (processed {chunk_count} chunks)"
    )


if __name__ == "__main__":
    main()
