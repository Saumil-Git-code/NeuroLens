"""
NeuroLens Model Utilities
=========================
Defines model architectures, device management, inference transforms,
and target layers for Grad-CAM explainability.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from torchvision import models, transforms

from ml import CLASS_NAMES

# Hardware device detection: uses GPU (CUDA) when available, fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet standard normalization constants (mean and std for RGB channels)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Grad-CAM target layer names mapped by architecture
GRADCAM_TARGETS: Dict[str, str] = {
    "vgg16": "features[-1]",
    "resnet18": "layer4[-1]",
    "efficientnet_b0": "features[-1]",
}

# Standard inference image transform pipeline:
# 1. Resize image to 224x224
# 2. Center crop to 224x224 to keep consistent spatial dimensions
# 3. Convert PIL Image to PyTorch Tensor [0, 1]
# 4. Normalize with ImageNet mean and std
TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model(arch: str = "vgg16", pretrained: bool = True, num_classes: int = len(CLASS_NAMES)) -> Tuple[nn.Module, str]:
    """
    Builds and returns a transfer learning model for brain tumor classification.

    Parameters
    ----------
    arch : str
        Architecture choice ('vgg16', 'resnet18', 'efficientnet_b0').
    pretrained : bool
        Whether to load pre-trained ImageNet weights.
    num_classes : int
        Number of output classes (default: 4 for glioma, meningioma, notumor, pituitary).

    Returns
    -------
    model : nn.Module
        PyTorch model with its final classifier modified to output `num_classes` logits.
    target_layer_name : str
        The string name of the target layer for Grad-CAM.
    """
    arch = arch.lower()
    weights_val = "DEFAULT" if pretrained else None

    if arch == "vgg16":
        # Load VGG-16 backbone
        model = models.vgg16(weights=weights_val)
        # Replace the final linear layer (Linear(4096, 1000) -> Linear(4096, 4))
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        target_layer_name = GRADCAM_TARGETS["vgg16"]

    elif arch == "resnet18":
        # Load ResNet-18 backbone
        model = models.resnet18(weights=weights_val)
        # Replace the final linear layer (Linear(512, 1000) -> Linear(512, 4))
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        target_layer_name = GRADCAM_TARGETS["resnet18"]

    elif arch == "efficientnet_b0":
        # Load EfficientNet-B0 backbone
        model = models.efficientnet_b0(weights=weights_val)
        # Replace the final linear layer in the classifier sequential (Linear(1280, 1000) -> Linear(1280, 4))
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        target_layer_name = GRADCAM_TARGETS["efficientnet_b0"]

    else:
        raise ValueError(
            f"Unsupported architecture '{arch}'. Supported models are: {list(GRADCAM_TARGETS.keys())}"
        )

    return model, target_layer_name


def get_gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    """
    Returns the target convolutional layer object for Grad-CAM explainability.

    Parameters
    ----------
    model : nn.Module
        The loaded PyTorch model instance.
    arch : str
        Architecture name ('vgg16', 'resnet18', or 'efficientnet_b0').

    Returns
    -------
    layer : nn.Module
        The PyTorch submodule layer to hook for Grad-CAM feature maps.
    """
    arch = arch.lower()
    if arch == "vgg16":
        # Last feature layer in VGG16
        return model.features[-1]
    elif arch == "resnet18":
        # Last bottleneck block in ResNet18
        return model.layer4[-1]
    elif arch == "efficientnet_b0":
        # Last feature block in EfficientNet-B0
        return model.features[-1]
    else:
        raise ValueError(
            f"Unsupported architecture '{arch}'. Supported models are: {list(GRADCAM_TARGETS.keys())}"
        )
