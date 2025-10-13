#!/usr/bin/env python3
"""Test script to verify Phi-3.5 model files can be loaded."""

import os
import sys
import torch
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def test_model_loading():
    """Test loading the local Phi-3.5 model."""
    model_path = project_root / "models" / "qgen_phi35"

    print(f"Testing model loading from: {model_path}")
    print(f"Model path exists: {model_path.exists()}")

    if not model_path.exists():
        print("ERROR: Model path does not exist!")
        return False

    # Check required files
    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"ERROR: Missing required files: {missing_files}")
        return False

    print("All required files present.")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        print("Tokenizer loaded successfully.")

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True,
        )
        print("Model loaded successfully.")

        # Test a simple inference
        print("Testing inference...")
        inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test inference successful: {response}")

        return True

    except Exception as e:
        print(f"ERROR during model loading/testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
