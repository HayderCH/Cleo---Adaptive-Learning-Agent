#!/usr/bin/env python3
"""Automated evaluation pipeline for model quality assessment."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import subprocess
import sys


class AutomatedEvaluator:
    """Run comprehensive evaluation suite."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "outputs" / "evaluations"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_unit_tests(self) -> Dict:
        """Run unit test suite."""
        print("🧪 Running unit tests...")
        start_time = time.time()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/",
                    "--tb=short",
                    "--json-report",
                    "--json-report-file=temp.json",
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            # Parse pytest JSON output
            json_file = self.project_root / "temp.json"
            if json_file.exists():
                with open(json_file) as f:
                    pytest_data = json.load(f)
                json_file.unlink()  # Clean up

                return {
                    "passed": pytest_data.get("summary", {}).get("passed", 0),
                    "failed": pytest_data.get("summary", {}).get("failed", 0),
                    "duration": time.time() - start_time,
                    "success": result.returncode == 0,
                }
            else:
                return {
                    "passed": 0,
                    "failed": 0,
                    "duration": time.time() - start_time,
                    "success": False,
                    "error": "Could not parse test results",
                }

        except Exception as e:
            return {
                "passed": 0,
                "failed": 0,
                "duration": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    def run_question_quality_eval(self) -> Dict:
        """Evaluate question generation quality."""
        print("📊 Evaluating question quality...")
        start_time = time.time()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_generator.py",
                    "--output",
                    str(self.results_dir / "quality_report.json"),
                    "--emit-quality-report",
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            # Read the generated report
            report_file = self.results_dir / "quality_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)

                return {
                    "duration": time.time() - start_time,
                    "success": result.returncode == 0,
                    "metrics": report,
                }
            else:
                return {
                    "duration": time.time() - start_time,
                    "success": False,
                    "error": "Quality report not generated",
                }

        except Exception as e:
            return {
                "duration": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    def run_performance_benchmarks(self) -> Dict:
        """Run performance benchmarks."""
        print("⚡ Running performance benchmarks...")
        start_time = time.time()

        try:
            # Test question generation performance
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import time
from services.qgen import generator
from services.qgen.schemas import GenerateRequest

req = GenerateRequest(
    subject='programming',
    focus_concepts=['python'],
    difficulty_target=0.5,
    bloom_target='understand',
    item_type='mcq',
    learner_snapshot={}
)

times = []
for i in range(5):
    start = time.time()
    question = generator.generate(req)
    times.append(time.time() - start)

print(f'Avg time: {sum(times)/len(times):.2f}s')
print(f'Min time: {min(times):.2f}s')
print(f'Max time: {max(times):.2f}s')
                """,
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Parse the output
                lines = result.stdout.strip().split("\n")
                metrics = {}
                for line in lines:
                    if "Avg time:" in line:
                        metrics["avg_generation_time"] = float(
                            line.split(":")[1].strip().rstrip("s")
                        )
                    elif "Min time:" in line:
                        metrics["min_generation_time"] = float(
                            line.split(":")[1].strip().rstrip("s")
                        )
                    elif "Max time:" in line:
                        metrics["max_generation_time"] = float(
                            line.split(":")[1].strip().rstrip("s")
                        )

                return {
                    "duration": time.time() - start_time,
                    "success": True,
                    "metrics": metrics,
                }
            else:
                return {
                    "duration": time.time() - start_time,
                    "success": False,
                    "error": result.stderr,
                }

        except Exception as e:
            return {
                "duration": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    def generate_report(self, results: Dict) -> Dict:
        """Generate comprehensive evaluation report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        report = {
            "timestamp": timestamp,
            "evaluation_results": results,
            "summary": {
                "overall_success": all(
                    r.get("success", False) for r in results.values()
                ),
                "total_duration": sum(r.get("duration", 0) for r in results.values()),
            },
        }

        # Save detailed report
        report_file = self.results_dir / f"evaluation_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation suite."""
        print("🚀 Starting full evaluation suite...")

        results = {
            "unit_tests": self.run_unit_tests(),
            "question_quality": self.run_question_quality_eval(),
            "performance": self.run_performance_benchmarks(),
        }

        report = self.generate_report(results)

        print("✅ Evaluation complete!")
        print(f"📄 Detailed report saved to: {self.results_dir}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Run automated evaluation suite")
    parser.add_argument(
        "--component",
        choices=["tests", "quality", "performance", "all"],
        default="all",
        help="Component to evaluate",
    )
    args = parser.parse_args()

    evaluator = AutomatedEvaluator(Path(__file__).parent.parent)

    if args.component == "tests":
        result = evaluator.run_unit_tests()
        print(json.dumps(result, indent=2))
    elif args.component == "quality":
        result = evaluator.run_question_quality_eval()
        print(json.dumps(result, indent=2))
    elif args.component == "performance":
        result = evaluator.run_performance_benchmarks()
        print(json.dumps(result, indent=2))
    else:
        result = evaluator.run_full_evaluation()
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
