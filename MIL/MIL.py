"""Aggregate per-class metrics from MIL validation and test predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def calculate_per_label_metrics(y_true, y_pred, y_score):
    """Calculate one-vs-rest metrics for every class in a prediction table."""
    num_classes = y_score.shape[1]
    labels = np.arange(num_classes)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total_samples = len(y_true)
    metrics = []

    for label in labels:
        true_positive = matrix[label, label]
        false_negative = matrix[label, :].sum() - true_positive
        false_positive = matrix[:, label].sum() - true_positive
        true_negative = total_samples - true_positive - false_negative - false_positive
        binary_true = y_true == label
        binary_pred = y_pred == label

        try:
            auc = roc_auc_score(binary_true, y_score[:, label])
        except ValueError as error:
            print(f"Unable to calculate AUC for class {label}: {error}")
            auc = np.nan

        metrics.append(
            {
                "label": int(label),
                "tp": int(true_positive),
                "fp": int(false_positive),
                "fn": int(false_negative),
                "tn": int(true_negative),
                "precision": round(precision_score(binary_true, binary_pred, zero_division=0), 4),
                "recall": round(recall_score(binary_true, binary_pred, zero_division=0), 4),
                "f1": round(f1_score(binary_true, binary_pred, zero_division=0), 4),
                "auc": round(auc, 4),
            }
        )

    return metrics


def process_csv_file(file_path):
    """Read a prediction CSV and return one metrics row per class."""
    frame = pd.read_csv(file_path)
    required_columns = {"label", "predicted_label"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")

    probability_columns = sorted(
        (column for column in frame.columns if column.startswith("prob_")),
        key=lambda column: int(column.removeprefix("prob_")),
    )
    if not probability_columns:
        raise ValueError("No probability columns named prob_<class_index> were found")

    return calculate_per_label_metrics(
        frame["label"].to_numpy(),
        frame["predicted_label"].to_numpy(),
        frame[probability_columns].to_numpy(),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-class metrics from Test_results_*.csv and Val_results_*.csv files."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing prediction CSV files")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input_dir>/mil_metrics.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    prediction_files = sorted(
        path
        for path in args.input_dir.rglob("*.csv")
        if path.name.startswith(("Test_results_", "Val_results_"))
    )
    rows = []
    for file_path in prediction_files:
        print(f"Processing {file_path}")
        try:
            file_metrics = process_csv_file(file_path)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            print(f"Skipping {file_path}: {error}")
            continue

        relative_path = file_path.relative_to(args.input_dir)
        split = "test" if file_path.name.startswith("Test") else "validation"
        for metrics in file_metrics:
            rows.append(
                {
                    "folder": str(relative_path.parent),
                    "file": file_path.name,
                    "split": split,
                    **metrics,
                }
            )

    if not rows:
        raise RuntimeError(f"No valid prediction CSV files found under {args.input_dir}")

    output_path = args.output or args.input_dir / "mil_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
