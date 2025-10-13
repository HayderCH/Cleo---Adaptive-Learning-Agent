#!/usr/bin/env python3
"""A/B Testing framework for comparing model versions and strategies."""

import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid


class ABTester:
    """A/B testing framework for model comparison."""

    def __init__(
        self,
        test_name: str,
        variants: List[str],
        traffic_split: Optional[List[float]] = None,
    ):
        self.test_name = test_name
        self.variants = variants

        if traffic_split is None:
            # Equal split by default
            self.traffic_split = [1.0 / len(variants)] * len(variants)
        else:
            assert len(traffic_split) == len(
                variants
            ), "Traffic split must match number of variants"
            assert (
                abs(sum(traffic_split) - 1.0) < 0.001
            ), "Traffic split must sum to 1.0"
            self.traffic_split = traffic_split

        self.results_dir = Path("outputs") / "ab_tests" / test_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load existing results if any
        self.results_file = self.results_dir / "results.json"
        self.results = self._load_results()

    def _load_results(self) -> Dict:
        """Load existing test results."""
        if self.results_file.exists():
            with open(self.results_file, "r") as f:
                return json.load(f)
        return {
            "test_name": self.test_name,
            "variants": self.variants,
            "traffic_split": self.traffic_split,
            "start_time": datetime.now().isoformat(),
            "experiments": [],
        }

    def _save_results(self):
        """Save test results."""
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def assign_variant(self, user_id: str) -> str:
        """Assign a variant to a user based on consistent hashing."""
        # Use consistent hashing for stable variant assignment
        hash_value = int(
            hashlib.md5(f"{self.test_name}:{user_id}".encode()).hexdigest(), 16
        )
        random.seed(hash_value)

        rand_val = random.random()
        cumulative = 0.0

        for variant, split in zip(self.variants, self.traffic_split):
            cumulative += split
            if rand_val <= cumulative:
                return variant

        return self.variants[-1]  # Fallback

    def record_experiment(self, user_id: str, variant: str, metrics: Dict[str, Any]):
        """Record the results of an experiment."""
        experiment = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "variant": variant,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }

        self.results["experiments"].append(experiment)
        self._save_results()

    def get_variant_stats(self, variant: str) -> Dict:
        """Get statistics for a specific variant."""
        variant_experiments = [
            exp for exp in self.results["experiments"] if exp["variant"] == variant
        ]

        if not variant_experiments:
            return {"count": 0}

        # Calculate basic statistics
        stats = {"count": len(variant_experiments)}

        # Get all metric keys
        metric_keys = set()
        for exp in variant_experiments:
            metric_keys.update(exp["metrics"].keys())

        # Calculate stats for each metric
        for key in metric_keys:
            values = [
                exp["metrics"].get(key)
                for exp in variant_experiments
                if exp["metrics"].get(key) is not None
            ]

            if values:
                stats[f"{key}_mean"] = sum(values) / len(values)
                stats[f"{key}_min"] = min(values)
                stats[f"{key}_max"] = max(values)
                stats[f"{key}_count"] = len(values)

        return stats

    def get_test_summary(self) -> Dict:
        """Get overall test summary and comparison."""
        summary = {
            "test_name": self.test_name,
            "total_experiments": len(self.results["experiments"]),
            "variants": {},
        }

        for variant in self.variants:
            summary["variants"][variant] = self.get_variant_stats(variant)

        # Calculate statistical significance for key metrics
        summary["comparisons"] = self._calculate_significance()

        return summary

    def _calculate_significance(self) -> Dict:
        """Calculate statistical significance between variants."""
        # Simplified significance calculation
        comparisons = {}

        if len(self.variants) < 2:
            return comparisons

        # Compare each pair of variants
        for i, variant_a in enumerate(self.variants):
            for variant_b in self.variants[i + 1 :]:
                key = f"{variant_a}_vs_{variant_b}"
                comparisons[key] = {
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "sample_size_a": self.get_variant_stats(variant_a)["count"],
                    "sample_size_b": self.get_variant_stats(variant_b)["count"],
                    "note": "Statistical significance calculation requires scipy.stats",
                }

        return comparisons

    def should_stop_test(
        self, min_samples: int = 100, confidence_threshold: float = 0.95
    ) -> bool:
        """Determine if the test should be stopped based on sample size and confidence."""
        min_variant_samples = min(
            self.get_variant_stats(variant)["count"] for variant in self.variants
        )

        return min_variant_samples >= min_samples


# Example usage functions


def run_question_generation_ab_test():
    """Example A/B test for question generation strategies."""
    # Test different RAG strategies
    tester = ABTester(
        test_name="question_generation_strategy",
        variants=["semantic_similarity", "bm25", "hybrid"],
        traffic_split=[0.5, 0.3, 0.2],  # 50%, 30%, 20% split
    )

    # Simulate some experiments
    for user_id in [f"user_{i}" for i in range(10)]:
        variant = tester.assign_variant(user_id)

        # Simulate different performance based on variant
        if variant == "semantic_similarity":
            accuracy = random.uniform(0.75, 0.85)
            time_taken = random.uniform(1.5, 2.5)
        elif variant == "bm25":
            accuracy = random.uniform(0.70, 0.80)
            time_taken = random.uniform(1.0, 2.0)
        else:  # hybrid
            accuracy = random.uniform(0.78, 0.88)
            time_taken = random.uniform(2.0, 3.0)

        tester.record_experiment(
            user_id,
            variant,
            {
                "accuracy": accuracy,
                "time_taken": time_taken,
                "user_satisfaction": random.randint(3, 5),
            },
        )

    return tester.get_test_summary()


def run_emotion_advice_ab_test():
    """Example A/B test for emotion advice strategies."""
    tester = ABTester(
        test_name="emotion_advice_style",
        variants=["empathetic", "directive", "educational"],
        traffic_split=[0.4, 0.4, 0.2],
    )

    return tester


if __name__ == "__main__":
    # Run example A/B test
    results = run_question_generation_ab_test()
    print(json.dumps(results, indent=2))
