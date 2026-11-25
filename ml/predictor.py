"""
NeuroLens Predictor
===================
Handles MRI scan inference with real PyTorch model weights or deterministic
mock fallback for local development and testing.

Includes temperature-scaled confidence calibration and Grad-CAM visual heatmaps.
"""

import hashlib
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import warnings

import numpy as np
from PIL import Image
import torch

from ml import CLASS_NAMES
from ml.model_utils import DEVICE, TRANSFORM, build_model, get_gradcam_layer

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    GradCAM = None
    show_cam_on_image = None
    ClassifierOutputTarget = None

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "ml" / "weights" / "best_model.pth"
MEDIA_ROOT = PROJECT_ROOT / "media"
HEATMAPS_DIR = MEDIA_ROOT / "heatmaps"

# Module-level cached model variables (lazy loaded, never at import time)
_model: Optional[torch.nn.Module] = None
_model_info: Optional[Dict[str, Any]] = None


def load_model(weights_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """
    Lazily loads the PyTorch model checkpoint.

    Parameters
    ----------
    weights_path : Path or str, optional
        Custom path to model weights (.pth). Defaults to ml/weights/best_model.pth.

    Returns
    -------
    dict or None
        Model info dictionary containing model, arch, class_names, and temperature,
        or None if weights are not available or loading fails.
    """
    global _model, _model_info

    # Return cached model if already loaded
    if _model is not None and _model_info is not None:
        return _model_info

    target_path = Path(weights_path) if weights_path else WEIGHTS_PATH

    # If the default best_model.pth doesn't exist, check for arch-specific weights
    if not target_path.exists():
        fallback_vgg = PROJECT_ROOT / "ml" / "weights" / "vgg16.pth"
        if fallback_vgg.exists():
            target_path = fallback_vgg
        else:
            return None

    try:
        # Load checkpoint dictionary from disk
        checkpoint = torch.load(target_path, map_location=DEVICE, weights_only=False)

        # Extract metadata and state dict
        arch = checkpoint.get("arch", "vgg16")
        class_names = checkpoint.get("class_names", CLASS_NAMES)
        state_dict = checkpoint.get("state_dict", checkpoint)
        temperature = float(checkpoint.get("temperature", 1.0))

        # Reconstruct model architecture and load weights
        model, _ = build_model(arch=arch, pretrained=False, num_classes=len(class_names))
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        # Cache in module-level variables
        _model = model
        _model_info = {
            "model": model,
            "arch": arch,
            "class_names": class_names,
            "temperature": temperature,
            "weights_path": str(target_path),
        }
        return _model_info

    except Exception as e:
        warnings.warn(f"Warning: Failed to load model weights from {target_path}: {e}")
        _model = None
        _model_info = None
        return None


def predict_real(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Runs real inference using the loaded PyTorch model and generates a Grad-CAM heatmap.

    Parameters
    ----------
    image_path : str or Path
        Path to the MRI image file on disk.

    Returns
    -------
    dict
        Prediction results including label, calibrated confidence, per-class probabilities,
        relative heatmap path, model architecture name, and is_mock flag.
    """
    global _model, _model_info

    # Ensure model is loaded
    if _model is None or _model_info is None:
        info = load_model()
        if info is None:
            raise RuntimeError(
                f"Model weights not found at {WEIGHTS_PATH}. Cannot execute predict_real."
            )

    model = _model_info["model"]
    arch = _model_info["arch"]
    temperature = float(_model_info.get("temperature", 1.0))
    class_names = _model_info.get("class_names", CLASS_NAMES)

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at {img_path}")

    # Load and preprocess image
    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    # Model inference with temperature scaling
    with torch.no_grad():
        logits = model(input_tensor)
        # Apply temperature scaling: calibrated_probs = softmax(logits / T)
        scaled_logits = logits / temperature
        probs_tensor = torch.softmax(scaled_logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs_tensor).item())
        confidence = float(probs_tensor[pred_idx].item())
        predicted_label = class_names[pred_idx]
        probs_dict = {
            cls_name: round(float(probs_tensor[i].item()), 4)
            for i, cls_name in enumerate(class_names)
        }

    # Generate Grad-CAM heatmap
    heatmap_path = None
    try:
        HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        heatmap_filename = f"scan_{timestamp}.png"
        heatmap_full_path = HEATMAPS_DIR / heatmap_filename

        target_layer = get_gradcam_layer(model, arch)

        # Prepare 224x224 RGB image normalized to [0, 1] for overlay
        resized_pil = img_pil.resize((224, 224))
        rgb_normalized = np.float32(resized_pil) / 255.0

        if GradCAM is not None:
            targets = [ClassifierOutputTarget(pred_idx)] if ClassifierOutputTarget is not None else None
            with GradCAM(model=model, target_layers=[target_layer]) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_normalized, grayscale_cam, use_rgb=True)
            Image.fromarray(cam_image).save(heatmap_full_path)
            heatmap_path = f"heatmaps/{heatmap_filename}"
    except Exception as e:
        warnings.warn(f"Warning: Grad-CAM generation failed ({e}). Proceeding without heatmap.")
        heatmap_path = None

    return {
        "label": predicted_label,
        "confidence": round(confidence, 4),
        "probabilities": probs_dict,
        "heatmap_path": heatmap_path,
        "model_name": arch,
        "is_mock": False,
    }


def predict_mock(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Deterministic mock predictor using MD5 image hash as random seed.
    Guarantees reproducible outputs for identical images while simulating realistic
    clinical confidence scores (~40% < 0.80 for the doctor review queue).

    Parameters
    ----------
    image_path : str or Path
        Path to the MRI image file.

    Returns
    -------
    dict
        Prediction dictionary matching the real inference format.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at {img_path}")

    # Read image bytes and compute MD5 hash
    with open(img_path, "rb") as f:
        file_bytes = f.read()
    img_hash = hashlib.md5(file_bytes).hexdigest()

    # Seed random generator with first 8 hex characters for deterministic behaviour
    seed = int(img_hash[:8], 16)
    rng = np.random.RandomState(seed)

    # Pick dominant class based on next 2 hex characters
    dominant_idx = int(img_hash[8:10], 16) % len(CLASS_NAMES)

    # Plausible confidence generation:
    # 40% of cases yield confidence < 0.80 so the physician review queue has items
    roll = int(img_hash[10:12], 16) % 100
    if roll < 40:
        dominant_prob = float(rng.uniform(0.55, 0.79))
    else:
        dominant_prob = float(rng.uniform(0.82, 0.98))

    # Distribute remaining probability mass across the other 3 classes
    remaining = 1.0 - dominant_prob
    other_weights = rng.dirichlet(np.ones(len(CLASS_NAMES) - 1)) * remaining

    probs_array = np.zeros(len(CLASS_NAMES), dtype=float)
    other_idx = 0
    for i in range(len(CLASS_NAMES)):
        if i == dominant_idx:
            probs_array[i] = dominant_prob
        else:
            probs_array[i] = float(other_weights[other_idx])
            other_idx += 1

    # Normalize to ensure exact sum of 1.0
    probs_array = probs_array / np.sum(probs_array)

    probabilities = {
        cls_name: round(float(probs_array[i]), 4)
        for i, cls_name in enumerate(CLASS_NAMES)
    }
    predicted_label = CLASS_NAMES[dominant_idx]
    confidence = round(float(probs_array[dominant_idx]), 4)

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "heatmap_path": None,
        "model_name": "mock",
        "is_mock": True,
    }


def get_prediction(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Main prediction entrypoint for the NeuroLens web application.
    Attempts to load real model weights; gracefully falls back to mock prediction
    if no trained model exists or if an inference error occurs.

    Parameters
    ----------
    image_path : str or Path
        Path to the uploaded brain MRI scan.

    Returns
    -------
    dict
        Prediction results dictionary.
    """
    model_info = load_model()

    if model_info is not None:
        try:
            return predict_real(image_path)
        except Exception as e:
            warnings.warn(f"Real prediction encountered an error: {e}. Falling back to mock predictor.")
            return predict_mock(image_path)

    return predict_mock(image_path)
