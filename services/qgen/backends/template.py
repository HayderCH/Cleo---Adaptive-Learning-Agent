from __future__ import annotations

import random
from typing import Dict, List

from services.qgen.backends.base import BaseQuestionGenerator
from services.qgen.schemas import GenerateRequest, Question, QuestionMeta


STEMS: Dict[str, List[str]] = {
    "remember": [
        "Define {concept} in one sentence.",
        "State the formula for {concept}.",
    ],
    "understand": [
        "Explain {concept} to a friend in simple terms.",
        "Which statement best describes {concept}?",
    ],
    "apply": [
        "Use {concept} to solve a basic example.",
        "Which step applies {concept} correctly?",
    ],
    "analyze": [
        "Identify which part depends on {concept} and why.",
        "Break down the process that uses {concept}.",
    ],
    "evaluate": [
        "Choose the best approach using {concept} and justify.",
        "Which argument for {concept} is strongest?",
    ],
    "create": [
        "Design a small example demonstrating {concept}.",
        "Propose a problem that requires {concept} to solve.",
    ],
}


FACT_BANK: Dict[str, Dict[str, Dict[str, List[str] | str]]] = {
    "linear equations": {
        "remember": {
            "correct": (
                "The slope-intercept form is y = mx + b, with m as slope "
                "and b as the y-intercept."
            ),
            "distractors": [
                "The quadratic formula is x = [-b ± √(b² - 4ac)] / (2a), "
                "which solves second-degree equations.",
                "A linear inequality describes a region using < or > "
                "instead of an equality sign.",
                "An exponential function follows y = a · b^x and grows "
                "multiplicatively with x.",
            ],
            "explanation": (
                "Slope-intercept form characterizes any straight line using "
                "its slope and intercept."
            ),
        },
        "understand": {
            "correct": (
                "A solution to a linear equation is any (x, y) pair that "
                "keeps both sides equal after substitution."
            ),
            "distractors": [
                "Solutions to quadratic equations can include up to two "
                "distinct roots satisfying ax² + bx + c = 0.",
                "Every linear equation must pass through the origin "
                "regardless of its slope.",
                "To solve a system of equations you only need to guess points "
                "until one works.",
            ],
            "explanation": (
                "Checking a solution means verifying the equality holds, not "
                "just approximating it."
            ),
        },
    },
    "gradient descent": {
        "remember": {
            "correct": (
                "Gradient descent updates parameters by subtracting the "
                "learning rate multiplied by the gradient."
            ),
            "distractors": [
                "Gradient descent adds the gradient to climb to the maximum "
                "of the loss function.",
                "Gradient descent keeps parameters fixed until the loss stops "
                "changing.",
                "Gradient descent resets parameters randomly whenever the "
                "loss increases.",
            ],
            "explanation": (
                "The gradient points toward steepest ascent, so we move in "
                "the opposite direction to descend."
            ),
        }
    },
    "momentum": {
        "remember": {
            "correct": (
                "Momentum scales the previous update and adds it to the "
                "current gradient step to smooth progress."
            ),
            "distractors": [
                "Momentum discards all past gradients to focus only on the "
                "newest direction.",
                "Momentum freezes weights whenever gradients oscillate.",
                "Momentum blindly increases the learning rate every step.",
            ],
            "explanation": (
                "Accumulating velocity keeps updates moving through shallow "
                "valleys efficiently."
            ),
        }
    },
    "dropout": {
        "remember": {
            "correct": (
                "Dropout randomly deactivates a subset of neurons during "
                "training to reduce co-adaptation."
            ),
            "distractors": [
                "Dropout permanently deletes neurons from the network.",
                "Dropout forces every neuron to activate in each batch.",
                "Dropout is a preprocessing technique for normalizing inputs.",
            ],
            "explanation": (
                "By omitting units temporarily, the network learns redundant "
                "representations and avoids overfitting."
            ),
        }
    },
    "bayes theorem": {
        "remember": {
            "correct": ("Bayes' theorem states P(A|B) = [P(B|A) · P(A)] / P(B)."),
            "distractors": [
                "Bayes' rule says P(A) = P(B) + P(A|B).",
                "Bayes' theorem claims conditional probabilities equal joint "
                "probabilities.",
                "Bayes' formula only applies when events are mutually " "exclusive.",
            ],
            "explanation": (
                "Bayes' theorem updates the probability of a hypothesis after "
                "observing evidence."
            ),
        }
    },
}


class TemplateQuestionGenerator(BaseQuestionGenerator):
    """Maintains the legacy rule-based question generation used for testing."""

    def _pick_concept(self, request: GenerateRequest) -> str:
        if request.focus_concepts:
            return random.choice(request.focus_concepts)
        return request.subject

    def _lookup_fact(
        self,
        concept: str,
        bloom: str,
    ) -> Dict[str, List[str] | str]:
        concept_key = concept.lower()
        concept_bank = FACT_BANK.get(concept_key, {})
        fact = concept_bank.get(bloom) or concept_bank.get("remember")
        if fact:
            return fact
        return {
            "correct": (
                f"A correct statement about {concept} precisely reflects its "
                "definition or standard usage."
            ),
            "distractors": [
                f"This option confuses {concept} with an unrelated idea.",
                f"This option mentions {concept} but omits the crucial " "detail.",
                f"This option misuses {concept} and would lead to an "
                "incorrect solution.",
            ],
            "explanation": (
                f"Select the description that aligns with how {concept} is "
                "applied in study or practice."
            ),
        }

    def _build_mcq(
        self,
        stem_template: str,
        concept: str,
        bloom: str,
    ) -> Question:
        stem = stem_template.format(concept=concept)
        fact = self._lookup_fact(concept, bloom)
        correct = str(fact["correct"])
        distractors = list(fact["distractors"])
        choices: List[str] = [correct, *distractors[:3]]
        random.shuffle(choices)
        correct_index = choices.index(correct)
        return Question(
            stem=stem,
            choices=choices,
            correct_index=correct_index,
            answer_key=correct,
            answer_expl=str(fact.get("explanation", "")),
            meta=QuestionMeta(
                concepts=[concept],
                difficulty_est=0.5,
                bloom="understand",
                source="template",
                version="v0",
            ),
        )

    def generate(self, request: GenerateRequest) -> Question:
        random.seed()
        concept = self._pick_concept(request)
        stems = STEMS.get(request.bloom_target, STEMS["understand"])
        stem_template = random.choice(stems)

        if request.item_type == "mcq":
            question = self._build_mcq(
                stem_template,
                concept,
                request.bloom_target,
            )
            question.meta.bloom = request.bloom_target
            question.meta.difficulty_est = request.difficulty_target
            return question

        stem = stem_template.format(concept=concept)
        fact = self._lookup_fact(concept, request.bloom_target)
        return Question(
            stem=stem,
            answer_key=str(
                fact.get(
                    "correct",
                    f"Provide a clear explanation or example involving " f"{concept}.",
                )
            ),
            answer_expl=str(
                fact.get(
                    "explanation",
                    f"Provide a clear explanation or example involving " f"{concept}.",
                )
            ),
            meta=QuestionMeta(
                concepts=[concept],
                difficulty_est=request.difficulty_target,
                bloom=request.bloom_target,
                source="template",
                version="v0",
            ),
        )
