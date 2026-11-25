"""
NeuroLens Model Training Script
===============================
Trains brain tumor classification backbones (VGG16, ResNet18, EfficientNet-B0)
using two-phase transfer learning on the Training dataset only.

Phases:
  1. Freeze backbone, train classification head only (epochs 1 to freeze_epochs)
  2. Unfreeze backbone with 10x lower learning rate for fine-tuning

Features:
  - Reproducible seeding
  - Deduplication filtering (reads ml/results/duplicates.txt)
  - Stratified train/val split
  - Mixed precision training with torch.amp on CUDA
  - Validation-based model checkpointing to ml/weights/{arch}.pth
  - Strict isolation: NEVER touches the Testing folder
"""

import argparse
import json
from pathlib import Path
import random
import time
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image
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


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_exclusion_set(dup_file: Path) -> Set[str]:
    """Reads duplicate file paths from ml/results/duplicates.txt to exclude from training."""
    exclusions = set()
    if not dup_file.exists():
        return exclusions

    with open(dup_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            # Store lowercase string of resolved path and filename for resilient matching
            exclusions.add(str(p.resolve()).lower())
            exclusions.add(p.name.lower())

    return exclusions


def collect_training_samples(
    train_dir: Path, exclusions: Set[str]
) -> Tuple[List[Tuple[Path, int]], List[Path]]:
    """
    Scans the Training directory subfolders for each tumor class.
    Excludes files identified in duplicates.txt.
    """
    samples: List[Tuple[Path, int]] = []
    excluded: List[Path] = []

    class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    for class_name in CLASS_NAMES:
        class_folder = train_dir / class_name
        if not class_folder.is_dir():
            print(f"Warning: Missing class directory: {class_folder}")
            continue

        for ext in VALID_EXTENSIONS:
            for img_path in list(class_folder.glob(f"*{ext}")) + list(class_folder.glob(f"*{ext.upper()}")):
                resolved_str = str(img_path.resolve()).lower()
                name_str = img_path.name.lower()

                if resolved_str in exclusions or name_str in exclusions:
                    excluded.append(img_path)
                else:
                    samples.append((img_path, class_to_idx[class_name]))

    # Deduplicate in case overlapping glob patterns matched
    unique_samples = []
    seen = set()
    for p, lbl in samples:
        if p not in seen:
            seen.add(p)
            unique_samples.append((p, lbl))

    return unique_samples, excluded


class MRIDataset(Dataset):
    """Custom PyTorch Dataset for MRI scans from a list of (image_path, label) tuples."""

    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def configure_backbone_freezing(model: nn.Module, arch: str, freeze: bool) -> None:
    """
    Freezes or unfreezes backbone feature extractor weights.
    When frozen, only the custom classification head parameters receive gradients.
    """
    arch = arch.lower()
    requires_grad = not freeze

    if arch == "vgg16":
        for param in model.features.parameters():
            param.requires_grad = requires_grad
    elif arch == "resnet18":
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = requires_grad
    elif arch == "efficientnet_b0":
        for param in model.features.parameters():
            param.requires_grad = requires_grad
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch across all batches."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    device_type = "cuda" if device.type == "cuda" else "cpu"

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision context
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates the model on the validation dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="NeuroLens MRI Transfer Learning Trainer")
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg16",
        choices=["vgg16", "resnet18", "efficientnet_b0"],
        help="Model architecture: 'vgg16' (primary), 'resnet18', or 'efficientnet_b0'.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="brain mri dataset",
        help="Path to dataset root folder containing Training/.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Total number of training epochs (default: 20).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: 16 for vgg16, 32 for resnet18/efficientnet_b0).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate for classification head (default: 0.001).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Target image size (default: 224x224).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of training dataset to reserve for validation (default: 0.15).",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=5,
        help="Epochs to train with backbone frozen before fine-tuning (default: 5).",
    )
    args = parser.parse_args()

    # 1. Reproducibility
    set_seed(args.seed)

    # Resolve batch size according to architecture defaults
    if args.batch_size is None:
        batch_size = 16 if args.arch == "vgg16" else 32
    else:
        batch_size = args.batch_size

    # Resolve filesystem paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    # Strictly enforce: train.py ONLY accesses the Training folder
    train_dir = data_dir / "Training"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found at: {train_dir}. Please check your data_dir path."
        )

    weights_dir = project_root / "ml" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir = project_root / "ml" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2. Check for deduplication exclusion list
    duplicates_file = results_dir / "duplicates.txt"
    exclusions = load_exclusion_set(duplicates_file)
    print("=" * 70)
    print("NEUROLENS MODEL TRAINING PIPELINE")
    print(f"Architecture:      {args.arch}")
    print(f"Device:            {DEVICE}")
    print(f"Dataset Training:  {train_dir}")
    print(f"Excluded Duplicates: {len(exclusions)} flagged images loaded from {duplicates_file.name}")
    print("=" * 70)

    # 3. Read training samples
    samples, excluded = collect_training_samples(train_dir, exclusions)
    if not samples:
        raise RuntimeError(f"No valid MRI images found in {train_dir}!")

    print(f"Active training samples:   {len(samples)}")
    print(f"Skipped duplicate images:  {len(excluded)}")

    # 4. Stratified train/val split
    labels = [label for _, label in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=args.val_split,
        stratify=labels,
        random_state=args.seed,
        shuffle=True,
    )
    print(f"Train split size:          {len(train_samples)}")
    print(f"Validation split size:     {len(val_samples)}")

    # 5. Data transforms
    train_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # 6. DataLoaders
    train_dataset = MRIDataset(train_samples, transform=train_transform)
    val_dataset = MRIDataset(val_samples, transform=val_transform)

    # Windows-safe multi-processing: num_workers=2
    is_cuda = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=is_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=is_cuda,
    )

    # 7. Model setup
    model, target_layer_name = build_model(arch=args.arch, pretrained=True, num_classes=len(CLASS_NAMES))
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Mixed precision scaler (enabled only when CUDA GPU is available)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    # 8. Phase 1: Freeze backbone, train classification head
    print(f"\n--- Phase 1: Freezing backbone for first {args.freeze_epochs} epochs ---")
    configure_backbone_freezing(model, args.arch, freeze=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = weights_dir / f"{args.arch}.pth"
    history = []

    start_time = time.time()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # 9. Phase 2: Unfreeze backbone after freeze_epochs and lower learning rate 10x
        if epoch == args.freeze_epochs + 1:
            print(f"\n--- Phase 2: Unfreezing backbone for fine-tuning (LR = {args.lr * 0.1}) ---")
            configure_backbone_freezing(model, args.arch, freeze=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)

        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, is_cuda, DEVICE
        )
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        epoch_time = time.time() - epoch_start

        phase_str = "Frozen" if epoch <= args.freeze_epochs else "Fine-tune"
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({phase_str}) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        # 13. Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                "arch": args.arch,
                "class_names": CLASS_NAMES,
                "state_dict": model.state_dict(),
                "val_accuracy": float(best_val_acc),
                "val_loss": float(val_loss),
                "epoch": best_epoch,
                "img_size": args.img_size,
            }
            torch.save(checkpoint, checkpoint_path)

            # If VGG16 (our primary web app model), also save as best_model.pth
            if args.arch == "vgg16":
                torch.save(checkpoint, weights_dir / "best_model.pth")

            print(f"  * New best validation accuracy: {best_val_acc * 100:.2f}% (Saved to {checkpoint_path.name})")

    total_duration = time.time() - start_time

    # 14. Save configuration and training summary
    config_path = results_dir / f"{args.arch}_config.json"
    run_config = {
        "arch": args.arch,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "learning_rate": args.lr,
        "freeze_epochs": args.freeze_epochs,
        "val_split": args.val_split,
        "seed": args.seed,
        "img_size": args.img_size,
        "device": str(DEVICE),
        "total_train_samples": len(train_samples),
        "total_val_samples": len(val_samples),
        "excluded_duplicates_count": len(excluded),
        "best_val_accuracy": round(float(best_val_acc), 4),
        "best_epoch": best_epoch,
        "training_duration_seconds": round(total_duration, 1),
        "weights_file": str(checkpoint_path),
        "history": history,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    # 16. Final printout
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Architecture:          {args.arch}")
    print(f"Best Validation Acc:   {best_val_acc * 100:.2f}% (at Epoch {best_epoch})")
    print(f"Saved Checkpoint:      {checkpoint_path}")
    print(f"Saved Config:          {config_path}")
    print(f"Total Training Time:   {total_duration / 60:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
