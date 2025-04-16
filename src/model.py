import segmentation_models_pytorch as smp
import torch.nn as nn 


def get_unet_model(encoder_name="resnet18", num_classes=10, encoder_weights="imagenet"):
    """
    Builds a U-Net model with the specified encoder.
    
    Args:
        encoder_name (str): Name of the encoder backbone (e.g., 'resnet18', 'resnet34', 'efficientnet-b0', etc.)
        num_classes (int): Number of output classes (default: 10 for SAR-RARP50)
        encoder_weights (str): Pretrained weights for encoder (e.g., 'imagenet', or None)

    Returns:
        model: segmentation model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes
    )
    return model