from __future__ import annotations

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


@dataclass
class EmotionalAdviceConfig:
    model_name: str = "models/qgen_phi35"  # Reuse the existing Phi model
    device_map: str = "auto"
    load_in_8bit: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    def __post_init__(self) -> None:
        # Resolve relative model paths to absolute paths (same logic as qgen transformer)
        if self.model_name and not self.model_name.startswith(
            ("http://", "https://", "/")
        ):
            # Check if it's a relative path that needs resolving
            if not os.path.isabs(self.model_name):
                # Get the project root (assuming this file is in services/emotion/)
                project_root = Path(__file__).resolve().parents[2]
                potential_path = project_root / self.model_name
                if potential_path.exists():
                    self.model_name = str(potential_path)
                    print(f"Resolved emotional advice model path to: {self.model_name}")
                else:
                    print(
                        f"Model path {potential_path} does not exist, using as-is: {self.model_name}"
                    )

    @classmethod
    def from_env(cls) -> EmotionalAdviceConfig:
        return cls(
            model_name=os.getenv("EMOTIONAL_ADVICE_MODEL", cls.model_name),
            device_map=os.getenv("EMOTIONAL_ADVICE_DEVICE_MAP", cls.device_map),
            load_in_8bit=os.getenv("EMOTIONAL_ADVICE_LOAD_IN_8BIT", "true").lower()
            == "true",
            max_new_tokens=int(
                os.getenv("EMOTIONAL_ADVICE_MAX_TOKENS", cls.max_new_tokens)
            ),
            temperature=float(
                os.getenv("EMOTIONAL_ADVICE_TEMPERATURE", cls.temperature)
            ),
            top_p=float(os.getenv("EMOTIONAL_ADVICE_TOP_P", cls.top_p)),
        )


class EmotionalAdviceAgent:
    """Generates personalized emotional support and advice based on emotion analysis."""

    SYSTEM_PROMPT = """You are an empathetic learning companion and emotional coach. Your role is to provide supportive, encouraging advice to help students navigate their learning emotions.

Based on the student's emotion analysis and their self-reported feelings, provide:
1. Empathetic acknowledgment of their current emotional state
2. 2-3 specific, actionable strategies to help them move forward
3. Encouraging words that validate their experience
4. Connection to their learning goals when appropriate

Keep your response warm, concise (under 150 words), and focused on emotional support rather than academic content. Be positive and solution-oriented."""

    def __init__(self, config: Optional[EmotionalAdviceConfig] = None):
        self.config = config or EmotionalAdviceConfig.from_env()
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _initialize_model(self):
        """Initialize by reusing the already loaded qgen generator pipeline."""
        if self._initialized:
            return

        try:
            # Import here to avoid circular imports
            from services.qgen.generator import _load_backend
            from services.qgen.backends.transformer import TransformerQuestionGenerator

            # Get the already loaded generator (cached)
            generator = _load_backend()

            if not isinstance(generator, TransformerQuestionGenerator):
                raise RuntimeError(
                    "QGen backend is not transformer. Please ensure transformer backend is configured."
                )

            # Get the pipeline from the generator
            pipeline = generator._ensure_pipeline()

            if pipeline is None or pipeline is False:
                raise RuntimeError(
                    "QGen transformer pipeline not available. Please ensure question generation is working first."
                )

            # Extract model and tokenizer from the pipeline
            self.model = pipeline.model
            self.tokenizer = pipeline.tokenizer

            self._initialized = True
            print("Successfully reused existing Phi model for emotional advice")

        except Exception as e:
            raise RuntimeError(f"Failed to reuse qgen transformer pipeline: {e}")

    def generate_advice(
        self,
        user_text: str,
        emotion_analysis: Dict[str, Any],
        emotion_state: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generate personalized emotional advice based on user input and emotion analysis.
        """
        self._initialize_model()

        # Format emotion context
        emotion_context = self._format_emotion_context(emotion_analysis, emotion_state)

        # Create the prompt
        prompt = f"""{self.SYSTEM_PROMPT}

Student's feelings: "{user_text}"

Emotion Analysis:
{emotion_context}

Please provide supportive advice:"""

        # Tokenize and generate
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and clean up the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the response
        if prompt in response:
            response = response[len(prompt) :].strip()

        # Clean up any remaining artifacts
        response = response.strip()
        if response.startswith("Please provide supportive advice:"):
            response = response[len("Please provide supportive advice:") :].strip()

        return response

    def _format_emotion_context(
        self,
        emotion_analysis: Dict[str, Any],
        emotion_state: Optional[Dict[str, float]],
    ) -> str:
        """Format emotion analysis data into readable context."""
        context_parts = []

        # Add flattened predictions if available
        if "flattened" in emotion_analysis and emotion_analysis["flattened"]:
            flattened = emotion_analysis["flattened"]
            if isinstance(flattened, list) and flattened:
                top_emotions = sorted(
                    flattened, key=lambda x: x.get("score", 0), reverse=True
                )[:3]
                emotion_strs = []
                for emo in top_emotions:
                    label = emo.get("label", "unknown")
                    score = emo.get("score", 0)
                    emotion_strs.append(f"{label}: {score:.2f}")
                context_parts.append(f"Top emotions: {', '.join(emotion_strs)}")

        # Add emotion state if available
        if emotion_state:
            state_parts = []
            for emotion, prob in emotion_state.items():
                if (
                    isinstance(prob, (int, float)) and prob > 0.1
                ):  # Only show significant emotions
                    state_parts.append(f"{emotion}: {prob:.2f}")
            if state_parts:
                context_parts.append(f"Current state: {', '.join(state_parts)}")

        # Add dominant bucket if available
        if (
            "dominant_bucket" in emotion_analysis
            and emotion_analysis["dominant_bucket"]
        ):
            bucket_name, bucket_score = emotion_analysis["dominant_bucket"]
            context_parts.append(f"Dominant affect: {bucket_name} ({bucket_score:.2f})")

        return (
            "\n".join(context_parts) if context_parts else "Emotion analysis available"
        )

    @classmethod
    def from_config(
        cls, config: Optional[EmotionalAdviceConfig] = None
    ) -> EmotionalAdviceAgent:
        """Create an EmotionalAdviceAgent from configuration."""
        return cls(config)


# Global instance for caching
_advice_agent_instance: Optional[EmotionalAdviceAgent] = None


def get_emotional_advice_agent() -> EmotionalAdviceAgent:
    """Get or create the global emotional advice agent instance."""
    global _advice_agent_instance
    if _advice_agent_instance is None:
        _advice_agent_instance = EmotionalAdviceAgent.from_config()
    return _advice_agent_instance
