"""
NeuroLens Model Evaluation and Calibration Script
=================================================
Performs rigorous validation-set temperature scaling and comprehensive
testing-set evaluation across clinical diagnostic metrics.

Workflow:
  1. Loads trained model checkpoint from ml/weights/{arch}.pth.
  2. STEP 1 (Temperature Scaling):
     - Recreates stratified validation split from Training/.
     - Fits temperature parameter T using Negative Log-Likelihood optimization.
     - Saves calibrated temperature back into checkpoint.
  3. STEP 2 (Testing Evaluation):
     - Loads unseen images from Testing/.
     - Computes Accuracy, Precision, Recall, F1, ROC-AUC, and ECE (before/after).
     - Benchmarks parameter count, disk size, and GPU/CPU inference latency.
     - Generates publication-ready Confusion Matrix and Reliability Diagrams.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers and CLI
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml import CLASS_NAMES
from ml.model_utils import (
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class MRIDataset(Dataset):
    """Simple PyTorch Dataset for evaluation."""

    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_exclusion_set(dup_file: Path) -> set:
    """Reads duplicate exclusions if available."""
    exclusions = set()
    if not dup_file.exists():
        return exclusions
    with open(dup_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            exclusions.add(str(p.resolve()).lower())
            exclusions.add(p.name.lower())
    return exclusions


def collect_images_from_folder(folder: Path, exclusions: set = None) -> List[Tuple[Path, int]]:
    """Recursively collects (image_path, label_idx) for each tumor class subdirectory."""
    if exclusions is None:
        exclusions = set()

    samples = []
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    for class_name in CLASS_NAMES:
        class_dir = folder / class_name
        if not class_dir.is_dir():
            continue
        for ext in VALID_EXTENSIONS:
            for p in list(class_dir.glob(f"*{ext}")) + list(class_dir.glob(f"*{ext.upper()}")):
                if str(p.resolve()).lower() not in exclusions and p.name.lower() not in exclusions:
                    samples.append((p, class_to_idx[class_name]))

    unique_samples = []
    seen = set()
    for p, lbl in samples:
        if p not in seen:
            seen.add(p)
            unique_samples.append((p, lbl))
    return unique_samples


def extract_logits_and_labels(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs forward pass on loader and returns all raw logits and ground truth labels."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(targets.numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def nll_objective(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Computes Negative Log-Likelihood of scaled logits given true labels."""
    scaled = logits / T
    scaled_max = np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled - scaled_max)
    log_probs = (scaled - scaled_max) - np.log(np.sum(exp_scaled, axis=1, keepdims=True))
    nll = -np.mean(log_probs[np.arange(len(labels)), labels])
    return float(nll)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE) and returns (ece, bin_confs, bin_accs).
    Divides confidence scores into n_bins and calculates weighted gap between confidence and accuracy.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_confs = []
    bin_accs = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        prop = np.mean(in_bin)

        if prop > 0:
            acc = float(np.mean(accuracies[in_bin]))
            conf = float(np.mean(confidences[in_bin]))
            ece += np.abs(acc - conf) * prop
            bin_accs.append(acc)
            bin_confs.append(conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((lower + upper) / 2.0)

    return float(ece), np.array(bin_confs), np.array(bin_accs)


def benchmark_inference_latency(
    model: nn.Module, img_size: int, device: torch.device
) -> float:
    """Measures single-image inference latency in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # 10 warm-up runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # 100 timed iterations
        num_runs = 100
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
        end = time.perf_counter()

    avg_ms = ((end - start) / num_runs) * 1000.0
    return float(avg_ms)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], save_path: Path, arch: str
) -> None:
    """Generates and saves a clean, annotated teal-themed confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Teal colormap gradient
    teal_cmap = LinearSegmentedColormap.from_list(
        "neuro_teal", ["#E0F2F1", "#80CBC4", "#26A69A", "#00897B", "#004D40"]
    )

    im = ax.imshow(cm, interpolation="nearest", cmap=teal_cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of MRI Scans", rotation=270, labelpad=15, fontsize=10)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=10, fontweight="bold")

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate counts inside cells
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            color = "white" if val > thresh else "#00332c"
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                color=color, fontsize=11, fontweight="bold"
            )

    ax.set_title(f"NeuroLens Confusion Matrix ({arch.upper()})", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("True Anatomical Class", fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_reliability_diagram(
    confs_before: np.ndarray,
    accs_before: np.ndarray,
    ece_before: float,
    confs_after: np.ndarray,
    accs_after: np.ndarray,
    ece_after: float,
    temperature: float,
    save_path: Path,
    arch: str,
) -> None:
    """Generates reliability diagram comparing model calibration before and after temperature scaling."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

    # Perfect calibration reference diagonal
    ax.plot([0, 1], [0, 1], "--", color="#757575", label="Perfect Calibration (y = x)", linewidth=1.5)

    # Uncalibrated curve
    ax.plot(
        confs_before,
        accs_before,
        marker="o",
        color="#E53935",
        label=f"Before Scaling (ECE: {ece_before:.4f})",
        linewidth=2,
    )

    # Temperature scaled calibrated curve
    ax.plot(
        confs_after,
        accs_after,
        marker="s",
        color="#00897B",
        label=f"After Scaling (T={temperature:.2f}, ECE: {ece_after:.4f})",
        linewidth=2,
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Confidence", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed Accuracy", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Reliability Diagram: Calibration Curve ({arch.upper()})",
        fontsize=12,
        pad=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="NeuroLens Model Evaluation and Calibration Tool")
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg16",
        choices=["vgg16", "resnet18", "efficientnet_b0"],
        help="Architecture to evaluate (default: 'vgg16').",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="brain mri dataset",
        help="Path to dataset root folder containing Training/ and Testing/.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Evaluation image resolution (default: 224).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    weights_dir = project_root / "ml" / "weights"
    results_dir = project_root / "ml" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = weights_dir / f"{args.arch}.pth"
    if not checkpoint_path.exists():
        # Fallback check for best_model.pth
        fallback = weights_dir / "best_model.pth"
        if fallback.exists():
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(
                f"Checkpoint file not found at {checkpoint_path}. Train the model first using train.py."
            )

    print("=" * 70)
    print("NEUROLENS COMPREHENSIVE EVALUATION & CALIBRATION")
    print(f"Architecture:      {args.arch}")
    print(f"Loading weights:   {checkpoint_path}")
    print(f"Device:            {DEVICE}")
    print("=" * 70)

    # 1. Load trained checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    class_names = checkpoint.get("class_names", CLASS_NAMES)

    # 2. Build model in eval mode
    model, _ = build_model(arch=args.arch, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # =========================================================================
    # STEP 1: Temperature Scaling on Validation Set (NEVER on Testing)
    # =========================================================================
    print("\n[STEP 1] Temperature Scaling on Validation Set...")
    train_dir = data_dir / "Training"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory required for validation split at: {train_dir}")

    duplicates_file = results_dir / "duplicates.txt"
    exclusions = load_exclusion_set(duplicates_file)
    train_samples = collect_images_from_folder(train_dir, exclusions)

    # Recreate the exact stratified split with seed 42 used during training
    labels_all = [lbl for _, lbl in train_samples]
    _, val_samples = train_test_split(
        train_samples,
        test_size=0.15,
        stratify=labels_all,
        random_state=42,
        shuffle=True,
    )

    val_dataset = MRIDataset(val_samples, transform=eval_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_logits, val_labels = extract_logits_and_labels(model, val_loader, DEVICE)

    # Optimize temperature T via NLL minimization
    res = minimize(
        nll_objective,
        x0=[1.0],
        args=(val_logits, val_labels),
        method="L-BFGS-B",
        bounds=[(0.05, 10.0)],
    )
    optimal_temperature = float(res.x[0]) if res.success else 1.0
    print(f"  -> Optimal Calibrated Temperature (T): {optimal_temperature:.4f}")

    # Save calibrated temperature into checkpoint
    checkpoint["temperature"] = optimal_temperature
    torch.save(checkpoint, checkpoint_path)
    if args.arch == "vgg16" and (weights_dir / "best_model.pth").exists():
        best_cp = torch.load(weights_dir / "best_model.pth", map_location="cpu", weights_only=False)
        best_cp["temperature"] = optimal_temperature
        torch.save(best_cp, weights_dir / "best_model.pth")
    print(f"  -> Checkpoint updated with temperature at {checkpoint_path.name}")

    # =========================================================================
    # STEP 2: Evaluation on Testing Set
    # =========================================================================
    print("\n[STEP 2] Running Evaluation on Testing Set...")
    test_dir = data_dir / "Testing"
    if not test_dir.exists():
        raise FileNotFoundError(f"Testing directory not found at: {test_dir}")

    test_samples = collect_images_from_folder(test_dir)
    print(f"Total testing images loaded: {len(test_samples)}")

    test_dataset = MRIDataset(test_samples, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    test_logits, test_labels = extract_logits_and_labels(model, test_loader, DEVICE)

    # Compute uncalibrated and calibrated probabilities
    raw_probs = softmax(test_logits, axis=1)
    calibrated_probs = softmax(test_logits / optimal_temperature, axis=1)

    preds = np.argmax(calibrated_probs, axis=1)

    # a. Test accuracy
    test_accuracy = float(np.mean(preds == test_labels))

    # b. Per-class precision, recall, F1
    clf_report = classification_report(
        test_labels, preds, target_names=class_names, output_dict=True, digits=4
    )

    # c. Confusion matrix
    cm = confusion_matrix(test_labels, preds)

    # d. Macro one-vs-rest ROC-AUC
    try:
        macro_auc = float(
            roc_auc_score(test_labels, calibrated_probs, multi_class="ovr", average="macro")
        )
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC ({e})")
        macro_auc = 0.0

    # e. ECE before and after calibration
    ece_before, confs_b, accs_b = compute_ece(raw_probs, test_labels, n_bins=15)
    ece_after, confs_a, accs_a = compute_ece(calibrated_probs, test_labels, n_bins=15)

    # f. Parameter count
    param_count = sum(p.numel() for p in model.parameters())

    # g. File size in MB
    file_size_mb = float(checkpoint_path.stat().st_size / (1024 * 1024))

    # h. Average inference time
    inference_ms = benchmark_inference_latency(model, args.img_size, DEVICE)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Architecture:              {args.arch}")
    print(f"Test Accuracy:             {test_accuracy * 100:.2f}%")
    print(f"Macro F1-Score:            {clf_report['macro avg']['f1-score']:.4f}")
    print(f"Macro ROC-AUC:             {macro_auc:.4f}")
    print(f"ECE (Before Scaling):      {ece_before:.4f}")
    print(f"ECE (After Scaling):       {ece_after:.4f} (Temp = {optimal_temperature:.2f})")
    print(f"Total Parameters:          {param_count:,}")
    print(f"Model File Size:           {file_size_mb:.2f} MB")
    print(f"Avg Inference Latency:     {inference_ms:.2f} ms ({DEVICE})")
    print("=" * 70)

    # 5. Save report JSON
    report_json_path = results_dir / f"{args.arch}_report.json"
    report_data = {
        "arch": args.arch,
        "test_accuracy": round(test_accuracy, 4),
        "macro_f1": round(float(clf_report["macro avg"]["f1-score"]), 4),
        "macro_auc": round(macro_auc, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "temperature": round(optimal_temperature, 4),
        "param_count": param_count,
        "file_size_mb": round(file_size_mb, 2),
        "inference_ms": round(inference_ms, 2),
        "device": str(DEVICE),
        "classification_report": clf_report,
        "confusion_matrix": cm.tolist(),
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    print(f"Report JSON saved to:       {report_json_path}")

    # 6. Save confusion matrix plot
    cm_plot_path = results_dir / f"{args.arch}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_plot_path, args.arch)
    print(f"Confusion Matrix saved to: {cm_plot_path}")

    # 7. Save reliability diagram
    rel_plot_path = results_dir / f"{args.arch}_reliability.png"
    plot_reliability_diagram(
        confs_b, accs_b, ece_before,
        confs_a, accs_a, ece_after,
        optimal_temperature, rel_plot_path, args.arch
    )
    print(f"Reliability Diagram saved: {rel_plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
