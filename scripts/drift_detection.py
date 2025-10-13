#!/usr/bin/env python3
"""Data drift detection for model inputs and performance monitoring."""

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from scipy import stats


class DataDriftDetector:
    """Detect data drift in model inputs and performance metrics."""

    def __init__(self, reference_window_days: int = 30, drift_threshold: float = 0.1):
        self.reference_window_days = reference_window_days
        self.drift_threshold = drift_threshold
        self.data_dir = Path("outputs") / "drift_detection"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Historical data storage
        self.question_patterns = defaultdict(list)
        self.emotion_patterns = defaultdict(list)
        self.performance_metrics = defaultdict(list)

    def record_question_data(self, question_data: Dict):
        """Record question generation data for drift detection."""
        timestamp = datetime.now()

        # Extract features for drift detection
        features = {
            "subject": question_data.get("subject", ""),
            "difficulty": question_data.get("difficulty_target", 0.5),
            "bloom_level": question_data.get("bloom_target", ""),
            "question_type": question_data.get("item_type", ""),
            "timestamp": timestamp.isoformat(),
        }

        # Store by day for aggregation
        day_key = timestamp.strftime("%Y-%m-%d")
        self.question_patterns[day_key].append(features)

        self._save_data()

    def record_emotion_data(self, emotion_data: Dict):
        """Record emotion analysis data for drift detection."""
        timestamp = datetime.now()

        features = {
            "emotion_distribution": emotion_data.get("emotion_distribution", {}),
            "confidence_scores": emotion_data.get("confidence_scores", []),
            "text_length": emotion_data.get("text_length", 0),
            "timestamp": timestamp.isoformat(),
        }

        day_key = timestamp.strftime("%Y-%m-%d")
        self.emotion_patterns[day_key].append(features)

        self._save_data()

    def record_performance_metric(self, metric_name: str, value: float):
        """Record performance metrics for drift detection."""
        timestamp = datetime.now()

        self.performance_metrics[metric_name].append(
            {"value": value, "timestamp": timestamp.isoformat()}
        )

        # Keep only recent data
        cutoff = datetime.now() - timedelta(days=self.reference_window_days * 2)
        self.performance_metrics[metric_name] = [
            m
            for m in self.performance_metrics[metric_name]
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]

        self._save_data()

    def detect_question_drift(self) -> Dict:
        """Detect drift in question generation patterns."""
        if len(self.question_patterns) < 7:  # Need at least a week of data
            return {"status": "insufficient_data"}

        # Get recent vs reference periods
        recent_days = self._get_recent_days(7)
        reference_days = self._get_reference_days(7)

        recent_data = []
        for day in recent_days:
            recent_data.extend(self.question_patterns.get(day, []))

        reference_data = []
        for day in reference_days:
            reference_data.extend(self.question_patterns.get(day, []))

        if not recent_data or not reference_data:
            return {"status": "insufficient_data"}

        # Check for distribution changes
        drift_signals = {}

        # Subject distribution drift
        recent_subjects = [d["subject"] for d in recent_data]
        ref_subjects = [d["subject"] for d in reference_data]

        if recent_subjects and ref_subjects:
            drift_signals["subject_distribution"] = self._calculate_distribution_drift(
                recent_subjects, ref_subjects
            )

        # Difficulty distribution drift
        recent_difficulty = [d["difficulty"] for d in recent_data]
        ref_difficulty = [d["difficulty"] for d in reference_data]

        drift_signals["difficulty_distribution"] = self._calculate_ks_drift(
            recent_difficulty, ref_difficulty
        )

        # Question type distribution drift
        recent_types = [d["question_type"] for d in recent_data]
        ref_types = [d["question_type"] for d in reference_data]

        drift_signals["question_type_distribution"] = (
            self._calculate_distribution_drift(recent_types, ref_types)
        )

        # Determine overall drift status
        max_drift = max(drift_signals.values()) if drift_signals else 0
        status = "high_drift" if max_drift > self.drift_threshold else "normal"

        return {
            "status": status,
            "max_drift_score": max_drift,
            "drift_signals": drift_signals,
            "recommendations": self._get_drift_recommendations(status, drift_signals),
        }

    def detect_performance_drift(self) -> Dict:
        """Detect drift in performance metrics."""
        drift_results = {}

        for metric_name, data_points in self.performance_metrics.items():
            if len(data_points) < 14:  # Need at least 2 weeks
                continue

            # Split into recent and reference periods
            midpoint = len(data_points) // 2
            recent_values = [p["value"] for p in data_points[midpoint:]]
            reference_values = [p["value"] for p in data_points[:midpoint]]

            if len(recent_values) >= 7 and len(reference_values) >= 7:
                drift_score = self._calculate_ks_drift(recent_values, reference_values)
                drift_results[metric_name] = {
                    "drift_score": drift_score,
                    "status": (
                        "high_drift" if drift_score > self.drift_threshold else "normal"
                    ),
                    "recent_mean": np.mean(recent_values),
                    "reference_mean": np.mean(reference_values),
                }

        return drift_results

    def _calculate_distribution_drift(self, recent: List, reference: List) -> float:
        """Calculate drift between categorical distributions."""
        if not recent or not reference:
            return 0.0

        # Simple Jaccard-like distance for categorical data
        recent_set = set(recent)
        ref_set = set(reference)

        intersection = len(recent_set & ref_set)
        union = len(recent_set | ref_set)

        if union == 0:
            return 0.0

        return 1.0 - (intersection / union)

    def _calculate_ks_drift(self, recent: List[float], reference: List[float]) -> float:
        """Calculate Kolmogorov-Smirnov drift for numerical data."""
        try:
            ks_statistic, _ = stats.ks_2samp(recent, reference)
            return ks_statistic
        except:
            return 0.0

    def _get_recent_days(self, days: int) -> List[str]:
        """Get list of recent day keys."""
        today = datetime.now()
        return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

    def _get_reference_days(self, days: int) -> List[str]:
        """Get list of reference period day keys."""
        today = datetime.now()
        start_ref = today - timedelta(days=self.reference_window_days)
        return [
            (start_ref - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)
        ]

    def _get_drift_recommendations(self, status: str, drift_signals: Dict) -> List[str]:
        """Generate recommendations based on drift detection."""
        recommendations = []

        if status == "high_drift":
            recommendations.append("Consider retraining models with recent data")

            if drift_signals.get("subject_distribution", 0) > self.drift_threshold:
                recommendations.append(
                    "Subject distribution has changed - review curriculum alignment"
                )

            if drift_signals.get("difficulty_distribution", 0) > self.drift_threshold:
                recommendations.append(
                    "Question difficulty patterns have shifted - recalibrate difficulty model"
                )

        recommendations.append("Monitor drift metrics weekly")
        recommendations.append("Collect more diverse training data if drift persists")

        return recommendations

    def _save_data(self):
        """Save drift detection data."""
        data = {
            "question_patterns": dict(self.question_patterns),
            "emotion_patterns": dict(self.emotion_patterns),
            "performance_metrics": dict(self.performance_metrics),
            "config": {
                "reference_window_days": self.reference_window_days,
                "drift_threshold": self.drift_threshold,
            },
        }

        with open(self.data_dir / "drift_data.json", "w") as f:
            json.dump(data, f, indent=2)


def main():
    """Example usage of drift detection."""
    detector = DataDriftDetector()

    # Simulate some data collection
    for i in range(20):
        # Simulate question data
        detector.record_question_data(
            {
                "subject": "programming" if i % 3 == 0 else "math",
                "difficulty_target": 0.3 + (i % 5) * 0.1,
                "bloom_target": "understand",
                "item_type": "mcq",
            }
        )

        # Simulate performance metrics
        detector.record_performance_metric("question_accuracy", 0.75 + (i % 10) * 0.01)
        detector.record_performance_metric("response_time", 2.0 + (i % 5) * 0.1)

    # Check for drift
    question_drift = detector.detect_question_drift()
    performance_drift = detector.detect_performance_drift()

    print("Question Drift Analysis:")
    print(json.dumps(question_drift, indent=2))
    print("\nPerformance Drift Analysis:")
    print(json.dumps(performance_drift, indent=2))


if __name__ == "__main__":
    main()
