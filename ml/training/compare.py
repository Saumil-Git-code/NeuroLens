"""
NeuroLens Model Architecture Comparison Tool
============================================
Aggregates and compares evaluation reports for all trained models
(e.g., VGG16, ResNet18, EfficientNet-B0).

Reads all ml/results/*_report.json files, outputs a formatted summary table
to the terminal, and exports results to ml/results/comparison.csv.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


COLUMNS = [
    ("arch", "Architecture"),
    ("accuracy", "Test Acc (%)"),
    ("macro_f1", "Macro F1"),
    ("macro_auc", "Macro AUC"),
    ("ece_after", "ECE (Calibrated)"),
    ("param_count", "Parameters"),
    ("file_size_mb", "Size (MB)"),
    ("inference_ms", "Latency (ms)"),
]


def load_reports(results_dir: Path) -> List[Dict[str, Any]]:
    """Finds and parses all *_report.json files in the results directory."""
    report_files = sorted(list(results_dir.glob("*_report.json")))
    reports = []

    for r_file in report_files:
        try:
            with open(r_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Map fields to canonical comparison row
                row = {
                    "arch": data.get("arch", r_file.stem.replace("_report", "")),
                    "accuracy": data.get("test_accuracy", 0.0),
                    "macro_f1": data.get("macro_f1", 0.0),
                    "macro_auc": data.get("macro_auc", 0.0),
                    "ece_after": data.get("ece_after", 0.0),
                    "param_count": data.get("param_count", 0),
                    "file_size_mb": data.get("file_size_mb", 0.0),
                    "inference_ms": data.get("inference_ms", 0.0),
                }
                reports.append(row)
        except Exception as e:
            print(f"Warning: Could not read {r_file.name}: {e}")

    return reports


def format_value(key: str, val: Any) -> str:
    """Formats values for clean tabular display."""
    if key == "accuracy":
        return f"{val * 100:.2f}%"
    elif key in ("macro_f1", "macro_auc", "ece_after"):
        return f"{val:.4f}"
    elif key == "param_count":
        return f"{val:,}"
    elif key in ("file_size_mb", "inference_ms"):
        return f"{val:.2f}"
    return str(val)


def print_table(reports: List[Dict[str, Any]]) -> None:
    """Prints a neat terminal table comparing all evaluated architectures."""
    headers = [label for _, label in COLUMNS]
    keys = [key for key, _ in COLUMNS]

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in reports:
        for idx, k in enumerate(keys):
            formatted_val = format_value(k, row[k])
            widths[idx] = max(widths[idx], len(formatted_val))

    # Add spacing padding
    widths = [w + 2 for w in widths]

    header_line = "|".join(h.center(w) for h, w in zip(headers, widths))
    sep_line = "|".join("-" * w for w in widths)

    print("\n" + "=" * len(header_line))
    print("NEUROLENS ARCHITECTURE COMPARISON BENCHMARK".center(len(header_line)))
    print("=" * len(header_line))
    print(header_line)
    print(sep_line)

    for row in reports:
        row_str = "|".join(
            (" " + format_value(k, row[k])).ljust(w) for k, w in zip(keys, widths)
        )
        print(row_str)

    print("=" * len(header_line) + "\n")


def export_csv(reports: List[Dict[str, Any]], csv_path: Path) -> None:
    """Saves the comparison table to a CSV file."""
    keys = [key for key, _ in COLUMNS]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in reports:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="NeuroLens Model Architecture Comparison")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="ml/results",
        help="Path to results directory containing *_report.json files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ml/results/comparison.csv",
        help="Destination path for the comparison CSV.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = project_root / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    reports = load_reports(results_dir)

    if not reports:
        print(f"\nNo evaluation reports found in {results_dir}.")
        print("To generate evaluation reports, run evaluate.py for your architectures, e.g.:")
        print("  python -m ml.training.evaluate --arch vgg16")
        print("  python -m ml.training.evaluate --arch resnet18")
        print("  python -m ml.training.evaluate --arch efficientnet_b0\n")
        return

    # Print formatted table to terminal
    print_table(reports)

    # Export to CSV
    export_csv(reports, output_csv)
    print(f"Comparison report exported successfully to: {output_csv}")


if __name__ == "__main__":
    main()
