import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

from src.inference import load_model, DEFAULT_MODEL_PATH, CLASS_NAMES
from src.multi_preprocessing import test_loader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_INDICES = list(range(len(CLASS_NAMES)))
NO_FIRE_CLASS = "no fire"
HIGH_CONFIDENCE_THRESHOLD = 0.8


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate_model(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict:
    """
    Evaluate the trained multiclass model on the full test dataset like an
    ML engineer would: overall/per-class metrics, where the confusion
    happens, how confident the model is when it's wrong, and the
    safety-critical miss/false-alarm rates that matter for a wildfire
    detector specifically.

    Returns:
        A dictionary containing overall, per-class, and error-analysis metrics.
    """

    logging.info("Loading trained model from %s", model_path)

    model = load_model(model_path)
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    confidences = []

    sample_paths = [path for path, _ in test_loader.dataset.samples]

    logging.info("Starting model evaluation on %d test batches", len(test_loader))

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            batch_confidence, predictions = torch.max(probabilities, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            confidences.extend(batch_confidence.cpu().numpy().tolist())

            logging.info(
                "Processed batch %d/%d",
                batch_index + 1,
                len(test_loader),
            )

    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    confidence_array = np.array(confidences)

    if len(sample_paths) != len(y_true_array):
        logging.warning(
            "Dataset sample count (%d) does not match prediction count (%d); "
            "misclassified report will omit file paths.",
            len(sample_paths),
            len(y_true_array),
        )
        sample_paths = [None] * len(y_true_array)

    overall_accuracy = accuracy_score(
        y_true_array,
        y_pred_array,
    )

    per_class_precision = precision_score(
        y_true_array,
        y_pred_array,
        labels=LABEL_INDICES,
        average=None,
        zero_division=0,
    )

    per_class_recall = recall_score(
        y_true_array,
        y_pred_array,
        labels=LABEL_INDICES,
        average=None,
        zero_division=0,
    )

    per_class_f1 = f1_score(
        y_true_array,
        y_pred_array,
        labels=LABEL_INDICES,
        average=None,
        zero_division=0,
    )

    per_class_support = np.bincount(y_true_array, minlength=len(CLASS_NAMES))

    weighted_precision = precision_score(
        y_true_array,
        y_pred_array,
        labels=LABEL_INDICES,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_true_array,
        y_pred_array,
        labels=LABEL_INDICES,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true_array,
        y_pred_array,
        labels=LABEL_INDICES,
        average="weighted",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=LABEL_INDICES,
    )

    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=LABEL_INDICES,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    confusion_pairs = analyze_confused_pairs(matrix)
    safety = analyze_safety_critical(y_true_array, y_pred_array)
    confidence_stats = analyze_confidence(y_true_array, y_pred_array, confidence_array)

    metrics = {
        "accuracy": overall_accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
        "confusion_matrix": matrix,
        "classification_report": report,
        "confusion_pairs": confusion_pairs,
        "safety": safety,
        "confidence_stats": confidence_stats,
    }
    metrics["insights"] = generate_insights(metrics)

    print_results(metrics)
    save_text_report(metrics)
    save_confusion_matrix(matrix)
    save_misclassified_report(y_true_array, y_pred_array, confidence_array, sample_paths)

    return metrics


def analyze_confused_pairs(matrix: np.ndarray, top_n: int = 5) -> list:
    """
    Rank (true_class -> predicted_class) mix-ups by how often they happen,
    so you can see exactly which classes the model confuses most.
    """

    pairs = []
    for true_index in range(matrix.shape[0]):
        for pred_index in range(matrix.shape[1]):
            if true_index == pred_index:
                continue
            count = int(matrix[true_index, pred_index])
            if count > 0:
                pairs.append(
                    {
                        "true_class": CLASS_NAMES[true_index],
                        "predicted_class": CLASS_NAMES[pred_index],
                        "count": count,
                    }
                )

    pairs.sort(key=lambda pair: pair["count"], reverse=True)
    return pairs[:top_n]


def analyze_safety_critical(y_true_array: np.ndarray, y_pred_array: np.ndarray) -> dict:
    """
    Accuracy alone hides the error type that matters most for a wildfire
    detector: a real fire/smoke/active-fire image predicted as "no fire"
    (a missed detection) is far more costly than a false alarm.
    """

    if NO_FIRE_CLASS not in CLASS_NAMES:
        return {}

    no_fire_index = CLASS_NAMES.index(NO_FIRE_CLASS)

    is_hazard_true = y_true_array != no_fire_index
    is_hazard_pred = y_pred_array != no_fire_index

    hazard_total = int(is_hazard_true.sum())
    missed_detections = int(np.sum(is_hazard_true & ~is_hazard_pred))
    miss_rate = missed_detections / hazard_total if hazard_total else 0.0

    no_fire_total = int((~is_hazard_true).sum())
    false_alarms = int(np.sum(~is_hazard_true & is_hazard_pred))
    false_alarm_rate = false_alarms / no_fire_total if no_fire_total else 0.0

    return {
        "hazard_total": hazard_total,
        "missed_detections": missed_detections,
        "miss_rate": miss_rate,
        "no_fire_total": no_fire_total,
        "false_alarms": false_alarms,
        "false_alarm_rate": false_alarm_rate,
    }


def analyze_confidence(
    y_true_array: np.ndarray,
    y_pred_array: np.ndarray,
    confidence_array: np.ndarray,
) -> dict:
    """
    A model that is wrong but unsure is a much smaller problem than a model
    that is wrong and confident. This flags the latter.
    """

    correct_mask = y_true_array == y_pred_array
    correct_confidence = confidence_array[correct_mask]
    incorrect_confidence = confidence_array[~correct_mask]

    high_confidence_errors = int(
        np.sum(incorrect_confidence >= HIGH_CONFIDENCE_THRESHOLD)
    )

    return {
        "mean_confidence_correct": float(correct_confidence.mean()) if correct_confidence.size else 0.0,
        "mean_confidence_incorrect": float(incorrect_confidence.mean()) if incorrect_confidence.size else 0.0,
        "total_errors": int((~correct_mask).sum()),
        "high_confidence_errors": high_confidence_errors,
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
    }


def generate_insights(metrics: dict) -> list:
    """
    Turn the raw numbers into the kind of "here's what's actually wrong and
    what to try next" notes an ML engineer would jot down during review.
    """

    insights = []

    f1_scores = metrics["per_class_f1"]
    weakest_index = int(np.argmin(f1_scores))
    insights.append(
        f"Weakest class is '{CLASS_NAMES[weakest_index]}' with F1={f1_scores[weakest_index]:.3f} "
        f"(precision={metrics['per_class_precision'][weakest_index]:.3f}, "
        f"recall={metrics['per_class_recall'][weakest_index]:.3f}, "
        f"support={int(metrics['per_class_support'][weakest_index])}). "
        "Consider collecting more training examples or targeted augmentation for this class."
    )

    if metrics["confusion_pairs"]:
        top_pair = metrics["confusion_pairs"][0]
        insights.append(
            f"Most common mistake: {top_pair['count']} '{top_pair['true_class']}' images were "
            f"predicted as '{top_pair['predicted_class']}'. These two classes likely look similar "
            "to the model -- consider hard-negative mining or reviewing labels for that pair."
        )

    safety = metrics.get("safety") or {}
    if safety:
        insights.append(
            f"Safety-critical miss rate: {safety['miss_rate'] * 100:.2f}% of real hazard images "
            f"({safety['missed_detections']}/{safety['hazard_total']}) were predicted as "
            f"'{NO_FIRE_CLASS}'. For a wildfire detector this is the most dangerous error type, "
            "even if it doesn't hurt overall accuracy much."
        )
        insights.append(
            f"False alarm rate: {safety['false_alarm_rate'] * 100:.2f}% of '{NO_FIRE_CLASS}' images "
            f"({safety['false_alarms']}/{safety['no_fire_total']}) were flagged as a hazard."
        )

    confidence_stats = metrics.get("confidence_stats") or {}
    if confidence_stats.get("total_errors"):
        insights.append(
            f"Mean confidence: {confidence_stats['mean_confidence_correct']:.3f} on correct "
            f"predictions vs {confidence_stats['mean_confidence_incorrect']:.3f} on incorrect ones. "
            f"{confidence_stats['high_confidence_errors']}/{confidence_stats['total_errors']} errors "
            f"were made at >={confidence_stats['high_confidence_threshold']:.0%} confidence -- the "
            "model was confidently wrong on these, so they're worth reviewing individually "
            "(see results/misclassified_examples.csv, sorted by confidence)."
        )

    return insights


def print_results(metrics: dict) -> None:
    """
    Print overall, per-class, and error-analysis results.
    """

    print("\n" + "=" * 60)
    print("MULTICLASS WILDFIRE MODEL EVALUATION")
    print("=" * 60)

    print(f"\nOverall Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")

    print("\nPer-Class Results")
    print("-" * 60)
    print(
        f"{'Class':<20}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1-Score':<12}"
        f"{'Support':<10}"
    )

    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name:<20}"
            f"{metrics['per_class_precision'][index]:<12.4f}"
            f"{metrics['per_class_recall'][index]:<12.4f}"
            f"{metrics['per_class_f1'][index]:<12.4f}"
            f"{int(metrics['per_class_support'][index]):<10}"
        )

    print("\nClassification Report")
    print("-" * 60)
    print(metrics["classification_report"])

    print("Confusion Matrix")
    print("-" * 60)
    print(metrics["confusion_matrix"])

    print("\nTop Confused Pairs (true -> predicted)")
    print("-" * 60)
    if metrics["confusion_pairs"]:
        for pair in metrics["confusion_pairs"]:
            print(f"{pair['true_class']} -> {pair['predicted_class']}: {pair['count']}")
    else:
        print("No misclassifications found.")

    safety = metrics.get("safety") or {}
    if safety:
        print("\nSafety-Critical Analysis")
        print("-" * 60)
        print(
            f"Missed detections (hazard predicted as '{NO_FIRE_CLASS}'): "
            f"{safety['missed_detections']}/{safety['hazard_total']} "
            f"({safety['miss_rate'] * 100:.2f}%)"
        )
        print(
            f"False alarms ('{NO_FIRE_CLASS}' predicted as hazard): "
            f"{safety['false_alarms']}/{safety['no_fire_total']} "
            f"({safety['false_alarm_rate'] * 100:.2f}%)"
        )

    confidence_stats = metrics.get("confidence_stats") or {}
    if confidence_stats:
        print("\nConfidence Analysis")
        print("-" * 60)
        print(f"Mean confidence (correct):   {confidence_stats['mean_confidence_correct']:.4f}")
        print(f"Mean confidence (incorrect): {confidence_stats['mean_confidence_incorrect']:.4f}")
        print(
            f"High-confidence errors (>= {confidence_stats['high_confidence_threshold']:.0%}): "
            f"{confidence_stats['high_confidence_errors']}/{confidence_stats['total_errors']}"
        )

    print("\nInsights & Suggested Next Steps")
    print("-" * 60)
    for insight in metrics["insights"]:
        print(f"- {insight}")


def save_text_report(metrics: dict) -> None:
    """
    Save the evaluation results to a text file.
    """

    report_path = RESULTS_DIR / "model_evaluation_report.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write("MULTICLASS WILDFIRE MODEL EVALUATION\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%\n")
        file.write(f"Weighted Precision: {metrics['weighted_precision']:.4f}\n")
        file.write(f"Weighted Recall: {metrics['weighted_recall']:.4f}\n")
        file.write(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}\n\n")

        file.write("Per-Class Results\n")
        file.write("-" * 60 + "\n")
        file.write(
            f"{'Class':<20}"
            f"{'Precision':<12}"
            f"{'Recall':<12}"
            f"{'F1-Score':<12}"
            f"{'Support':<10}\n"
        )

        for index, class_name in enumerate(CLASS_NAMES):
            file.write(
                f"{class_name:<20}"
                f"{metrics['per_class_precision'][index]:<12.4f}"
                f"{metrics['per_class_recall'][index]:<12.4f}"
                f"{metrics['per_class_f1'][index]:<12.4f}"
                f"{int(metrics['per_class_support'][index]):<10}\n"
            )

        file.write("\nClassification Report\n")
        file.write("-" * 60 + "\n")
        file.write(metrics["classification_report"])

        file.write("\nConfusion Matrix\n")
        file.write("-" * 60 + "\n")
        file.write(np.array2string(metrics["confusion_matrix"]))

        file.write("\n\nTop Confused Pairs (true -> predicted)\n")
        file.write("-" * 60 + "\n")
        if metrics["confusion_pairs"]:
            for pair in metrics["confusion_pairs"]:
                file.write(f"{pair['true_class']} -> {pair['predicted_class']}: {pair['count']}\n")
        else:
            file.write("No misclassifications found.\n")

        safety = metrics.get("safety") or {}
        if safety:
            file.write("\nSafety-Critical Analysis\n")
            file.write("-" * 60 + "\n")
            file.write(
                f"Missed detections (hazard predicted as '{NO_FIRE_CLASS}'): "
                f"{safety['missed_detections']}/{safety['hazard_total']} "
                f"({safety['miss_rate'] * 100:.2f}%)\n"
            )
            file.write(
                f"False alarms ('{NO_FIRE_CLASS}' predicted as hazard): "
                f"{safety['false_alarms']}/{safety['no_fire_total']} "
                f"({safety['false_alarm_rate'] * 100:.2f}%)\n"
            )

        confidence_stats = metrics.get("confidence_stats") or {}
        if confidence_stats:
            file.write("\nConfidence Analysis\n")
            file.write("-" * 60 + "\n")
            file.write(f"Mean confidence (correct):   {confidence_stats['mean_confidence_correct']:.4f}\n")
            file.write(f"Mean confidence (incorrect): {confidence_stats['mean_confidence_incorrect']:.4f}\n")
            file.write(
                f"High-confidence errors (>= {confidence_stats['high_confidence_threshold']:.0%}): "
                f"{confidence_stats['high_confidence_errors']}/{confidence_stats['total_errors']}\n"
            )

        file.write("\nInsights & Suggested Next Steps\n")
        file.write("-" * 60 + "\n")
        for insight in metrics["insights"]:
            file.write(f"- {insight}\n")

    logging.info("Saved text report to %s", report_path)


def save_confusion_matrix(matrix: np.ndarray) -> None:
    """
    Save both a raw-count and a row-normalized confusion matrix image.
    The normalized version makes per-class error rates readable even
    when classes are imbalanced.
    """

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(values_format="d", xticks_rotation=45, cmap="Blues")
    plt.title("Wildfire Model Confusion Matrix (Counts)")
    plt.tight_layout()

    output_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved confusion matrix image to %s", output_path)

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    normalized_display = ConfusionMatrixDisplay(
        confusion_matrix=normalized_matrix,
        display_labels=CLASS_NAMES,
    )
    normalized_display.plot(values_format=".0%", xticks_rotation=45, cmap="Blues")
    plt.title("Wildfire Model Confusion Matrix (Row-Normalized)")
    plt.tight_layout()

    normalized_output_path = RESULTS_DIR / "confusion_matrix_normalized.png"
    plt.savefig(normalized_output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved normalized confusion matrix image to %s", normalized_output_path)


def save_misclassified_report(
    y_true_array: np.ndarray,
    y_pred_array: np.ndarray,
    confidence_array: np.ndarray,
    sample_paths: list,
) -> None:
    """
    Save every misclassified example to a CSV, sorted by confidence
    (highest first) so the most confidently-wrong -- and most concerning --
    predictions are the first ones reviewed.
    """

    incorrect_indices = np.where(y_true_array != y_pred_array)[0]
    rows = sorted(
        (
            {
                "file_path": sample_paths[index],
                "true_class": CLASS_NAMES[y_true_array[index]],
                "predicted_class": CLASS_NAMES[y_pred_array[index]],
                "confidence": float(confidence_array[index]),
            }
            for index in incorrect_indices
        ),
        key=lambda row: row["confidence"],
        reverse=True,
    )

    report_path = RESULTS_DIR / "misclassified_examples.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["file_path", "true_class", "predicted_class", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logging.info(
        "Saved %d misclassified examples (sorted by confidence, worst first) to %s",
        len(rows),
        report_path,
    )


if __name__ == "__main__":
    evaluate_model()
