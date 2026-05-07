"""
Model Utilities for SABR Notebooks

This module contains model initialization and analysis functions for SABR notebooks.
"""

import torch
import torch.nn as nn
from torchvision import models

def get_vgg_model(num_classes=14, pretrained=True):
    """Get a VGG11 model with a custom classifier."""
    # Check if using newer torchvision where pretrained is replaced with weights
    try:
        model = models.vgg11(pretrained=pretrained)
    except:
        model = models.vgg11(weights='DEFAULT' if pretrained else None)
    
    # Replace the classifier with our custom sequence
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),     # Dimensionality reduction
        nn.ReLU(inplace=True),           # Non-linearity with memory optimization
        nn.Dropout(0.5),                 # Regularization during initial training
        nn.Linear(512, num_classes),     # Output layer
        nn.Softmax(dim=1)                # Probability distribution
    )
    
    return model

def get_mobilenet_model(num_classes=14, pretrained=True):
    """Get a MobileNetV2 model with a custom classifier."""
    # Check if using newer torchvision where pretrained is replaced with weights
    try:
        model = models.mobilenet_v2(pretrained=pretrained)
    except:
        model = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
    
    # Replace the classifier with our custom sequence
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, num_classes),
        nn.Softmax(dim=1)
    )
    
    return model

def print_model_structure(model, input_size=(3, 128, 128)):
    """
    Print the structure of a PyTorch model with parameter counts.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    # Detect device
    device = next(model.parameters()).device
    
    # Create a dummy input tensor on the same device as the model
    x = torch.randn(1, *input_size).to(device)
    
    # Create a dictionary to store layer outputs
    layer_outputs = {}
    
    # Hook to capture layer outputs
    def get_layer_output_hook(name):
        def hook(module, input, output):
            layer_outputs[name] = output
        return hook
    
    # Register hooks for all layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
            hook = module.register_forward_hook(get_layer_output_hook(name))
            hooks.append(hook)
    
    # Pass input through the model to get feature map sizes
    with torch.no_grad():  # Add no_grad for efficiency
        model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print model information
    print("=" * 80)
    print(f"Model Structure: {model.__class__.__name__} (on {device})")
    print("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    # Print layer information
    print(f"{'Layer Name':<40} {'Type':<15} {'Output Shape':<20} {'Parameters':<15}")
    print("-" * 90)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
            # Calculate parameters
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            total_params += params
            trainable_params += trainable
            
            # Get output shape
            output_shape = tuple(layer_outputs[name].shape) if name in layer_outputs else "-"
            
            # Print layer info
            print(f"{name:<40} {module.__class__.__name__:<15} {str(output_shape):<20} {params:,}")
    
    print("-" * 90)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 80)
    
    return total_params, trainable_params

def compare_models(vgg_model, mobilenet_model, input_size=(3, 128, 128)):
    """
    Compare VGG11 and MobileNetV2 models in terms of parameters and FLOPs.
    
    Args:
        vgg_model: VGG model to compare
        mobilenet_model: MobileNetV2 model to compare
        input_size: Input tensor size (C, H, W)
        
    Returns:
        Dictionary with comparison metrics
    """
    # Save original devices
    vgg_device = next(vgg_model.parameters()).device
    mobilenet_device = next(mobilenet_model.parameters()).device
    
    # IMPORTANT: We're doing comparison on CPU to avoid issues with tensors and models on different devices
    # Make temporary CPU copies of the models
    vgg_model_cpu = vgg_model.cpu()
    mobilenet_model_cpu = mobilenet_model.cpu()
    
    # Print comparison header
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    
    # Get model sizes
    print("Analyzing VGG11 structure...")
    vgg_params, vgg_trainable = print_model_structure(vgg_model_cpu, input_size)
    
    print("\nAnalyzing MobileNetV2 structure...")
    mobilenet_params, mobilenet_trainable = print_model_structure(mobilenet_model_cpu, input_size)
    
    # Calculate approximate FLOPs (simplified calculation)
    print("\nCalculating computational complexity...")
    vgg_flops = 0
    mobilenet_flops = 0
    
    # Create dummy input
    x = torch.randn(1, *input_size)
    
    # Simple FLOPs estimation for each layer (VGG)
    with torch.no_grad():
        for name, module in vgg_model_cpu.named_modules():
            if isinstance(module, nn.Conv2d):
                # Estimate Conv2d FLOPs
                out_h = input_size[1] // 2 if '3' in name or '8' in name or '13' in name else input_size[1]
                out_w = input_size[2] // 2 if '3' in name or '8' in name or '13' in name else input_size[2]
                vgg_flops += 2 * module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * out_h * out_w
            elif isinstance(module, nn.Linear):
                # Estimate Linear FLOPs
                vgg_flops += 2 * module.in_features * module.out_features
    
        # MobileNetV2 FLOPs
        for name, module in mobilenet_model_cpu.named_modules():
            if isinstance(module, nn.Conv2d):
                # Estimate Conv2d FLOPs
                # This is a very rough estimate for MobileNetV2
                out_h = input_size[1] // 2 if 'downsample' in name else input_size[1]
                out_w = input_size[2] // 2 if 'downsample' in name else input_size[2]
                mobilenet_flops += 2 * module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * out_h * out_w
            elif isinstance(module, nn.Linear):
                # Estimate Linear FLOPs
                mobilenet_flops += 2 * module.in_features * module.out_features
    
    # Format numbers for readability
    def format_number(num):
        if num >= 1e9:
            return f"{num/1e9:.2f}G"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        else:
            return f"{num}"
    
    # Print comparison
    print(f"{'Model':<15} {'Parameters':<15} {'Approx. FLOPs':<20} {'Relative Size':<15} {'Relative FLOPs':<15}")
    print("-" * 85)
    print(f"{'VGG11':<15} {vgg_params:,} ({format_number(vgg_params)}) {format_number(vgg_flops)} {'100%':<15} {'100%':<15}")
    print(f"{'MobileNetV2':<15} {mobilenet_params:,} ({format_number(mobilenet_params)}) {format_number(mobilenet_flops)} {mobilenet_params/vgg_params*100:.2f}% {mobilenet_flops/vgg_flops*100:.2f}%")
    print("=" * 85)
    
    # Move models back to their original devices
    vgg_model.to(vgg_device)
    mobilenet_model.to(mobilenet_device)
    print(f"\nModels restored to their original devices: VGG on {vgg_device}, MobileNet on {mobilenet_device}")
    print("Note: Comparison performed on CPU to avoid device mismatch issues.")
    
    # Return comparison metrics
    return {
        'vgg': {
            'params': vgg_params,
            'trainable_params': vgg_trainable,
            'flops': vgg_flops,
            'device': vgg_device
        },
        'mobilenet': {
            'params': mobilenet_params,
            'trainable_params': mobilenet_trainable,
            'flops': mobilenet_flops,
            'device': mobilenet_device
        },
        'comparison': {
            'size_ratio': mobilenet_params/vgg_params,
            'flops_ratio': mobilenet_flops/vgg_flops
        }
    }

def create_model_comparison_table(vgg_model, mobilenet_model, input_size=(3, 128, 128)):
    """
    Create a detailed comparison table between VGG and MobileNetV2 models.
    
    Args:
        vgg_model: VGG model to compare
        mobilenet_model: MobileNetV2 model to compare
        input_size: Input tensor size (C, H, W)
        
    Returns:
        DataFrame with comparison metrics
    """
    import pandas as pd
    
    # Get comparison metrics
    metrics = compare_models(vgg_model, mobilenet_model, input_size)
    
    # Create dataframe for comparison
    df = pd.DataFrame({
        'Metric': [
            'Total Parameters', 
            'Trainable Parameters',
            'Approximate FLOPs',
            'Memory Footprint (MB)',
            'Relative Size',
            'Relative Complexity'
        ],
        'VGG11': [
            f"{metrics['vgg']['params']:,}",
            f"{metrics['vgg']['trainable_params']:,}",
            f"{metrics['vgg']['flops']:,}",
            f"{metrics['vgg']['params'] * 4 / (1024 * 1024):.2f}",
            "100%",
            "100%"
        ],
        'MobileNetV2': [
            f"{metrics['mobilenet']['params']:,}",
            f"{metrics['mobilenet']['trainable_params']:,}",
            f"{metrics['mobilenet']['flops']:,}",
            f"{metrics['mobilenet']['params'] * 4 / (1024 * 1024):.2f}",
            f"{metrics['comparison']['size_ratio']*100:.2f}%",
            f"{metrics['comparison']['flops_ratio']*100:.2f}%"
        ]
    })
    
    return df
