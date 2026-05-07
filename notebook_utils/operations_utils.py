"""
Operations Calculation Utilities for SABR Notebooks

This module contains functions for calculating operations and computational savings
for models trained with SABR.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings

def calculate_operations(model, input_size=(3, 128, 128)):
    """
    Calculate the number of operations for each layer in the model.
    
    Args:
        model: The PyTorch model
        input_size: Input tensor size (C, H, W)
        
    Returns:
        Dictionary mapping layer names to operation counts
    """
    operations = {}
    total_ops = 0
    
    # For VGG11, define the input dimensions for each layer
    # These values are specific to VGG11 with input size 128x128
    vgg_input_heights = {
        'features.0': 128, # Conv2d
        'features.3': 64,  # Conv2d after MaxPool2d
        'features.6': 32,  # Conv2d after MaxPool2d
        'features.8': 32,  # Conv2d
        'features.11': 16, # Conv2d after MaxPool2d
        'features.13': 16, # Conv2d
        'features.16': 8,  # Conv2d after MaxPool2d
        'features.18': 8,  # Conv2d
        'classifier.0': 1, # Linear (fully connected)
        'classifier.3': 1  # Linear (fully connected)
    }
    
    # For MobileNetV2, define approximate input dimensions
    # These are based on typical MobileNetV2 architecture with 128x128 input
    mobilenet_input_heights = {
        'features.0.0': 128,  # First Conv2d
        'features.1.conv': 64,  # First bottleneck
        'features.2.conv': 32,  # Second bottleneck
        'features.3.conv': 32,  # Third bottleneck
        'features.4.conv': 16,  # Fourth bottleneck
        'features.5.conv': 16,  # Fifth bottleneck
        'features.6.conv': 8,   # Sixth bottleneck
        'features.7.conv': 8,   # Seventh bottleneck
        'features.8.0': 8,      # Last Conv2d
        'classifier.1': 1,      # Linear (fully connected)
    }
    
    # Determine if it's VGG or MobileNet based on first layer
    is_vgg = False
    is_mobilenet = False
    
    for name, _ in model.named_modules():
        if 'features.0' in name and not '0.0' in name:
            is_vgg = True
            break
        if 'features.0.0' in name:
            is_mobilenet = True
            break
    
    input_heights = vgg_input_heights if is_vgg else mobilenet_input_heights
    
    # Try a more accurate approach using hooks if possible
    try:
        device = next(model.parameters()).device
        x = torch.randn(1, *input_size).to(device)
        
        # Dictionary to store operations count from hooks
        hook_operations = {}
        
        # Registration hook to calculate MACs
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(module, nn.Conv2d):
                    # For Conv2d: out_channels * (in_channels/groups) * kernel_h * kernel_w * out_h * out_w
                    in_c = module.in_channels
                    out_c = module.out_channels
                    k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                    out_h, out_w = output.size()[2:]  # Get actual output dimensions
                    groups = module.groups
                    
                    # Calculate MACs (multiply-accumulate operations)
                    macs = out_c * (in_c / groups) * k_h * k_w * out_h * out_w
                    hook_operations[name] = int(macs)
                elif isinstance(module, nn.Linear):
                    # For Linear: out_features * in_features
                    macs = module.out_features * module.in_features
                    hook_operations[name] = int(macs)
            return hook
        
        # Register hooks for all Conv2d and Linear layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass to trigger hooks
        with torch.no_grad():
            _ = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Update operations dictionary
        operations.update(hook_operations)
        total_ops = sum(hook_operations.values())
        
    except Exception as e:
        warnings.warn(f"Dynamic operations calculation failed: {e}\nFalling back to static calculation method...")
        
        # Iterate through named modules using static approach as fallback
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Get parameters for convolutional layer
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                
                # Find the closest matching layer name in our input_heights dictionary
                matching_key = None
                for key in input_heights:
                    if name.startswith(key):
                        matching_key = key
                        break
                
                if matching_key:
                    input_height = input_heights[matching_key]
                    input_width = input_height  # Assuming square input
                    
                    # Calculate output dimensions (simplified, assuming padding preserves dimensions)
                    output_height = input_height
                    output_width = output_height
                    
                    # Calculate MACs for this layer
                    macs = in_channels * out_channels * kernel_size * kernel_size * output_height * output_width
                    operations[name] = macs
                    total_ops += macs
            
            elif isinstance(module, nn.Linear):
                # Get parameters for fully connected layer
                in_features = module.in_features
                out_features = module.out_features
                
                # Calculate MACs for this layer
                macs = in_features * out_features
                operations[name] = macs
                total_ops += macs
    
    operations['total'] = total_ops
    return operations

def calculate_pruned_operations(model, pruning_stats, input_size=(3, 128, 128)):
    """
    Calculate the number of operations for the pruned model.
    
    Args:
        model: The PyTorch model
        pruning_stats: Dictionary with pruning statistics for each layer
        input_size: Input tensor size (C, H, W)
        
    Returns:
        Dictionary with operation statistics for the pruned model
    """
    base_operations = calculate_operations(model, input_size=input_size)
    pruned_operations = {}
    total_pruned_ops = 0
    
    # Build a mapping between pruning stats and operations layers
    layer_mapping = {}
    
    # First, create a clean mapping between pruning stat layers and operation layers
    for pruning_layer in pruning_stats.keys():
        best_match = None
        best_match_len = 0
        
        for op_layer in base_operations.keys():
            if op_layer != 'total':
                # Try to find the best matching operation layer for each pruning layer
                # Either exact match or longest common prefix
                if pruning_layer == op_layer:
                    best_match = op_layer
                    break
                elif pruning_layer.startswith(op_layer) and len(op_layer) > best_match_len:
                    best_match = op_layer
                    best_match_len = len(op_layer)
                elif op_layer.startswith(pruning_layer) and len(pruning_layer) > best_match_len:
                    best_match = op_layer
                    best_match_len = len(pruning_layer)
        
        if best_match:
            layer_mapping[pruning_layer] = best_match
    
    # Calculate pruned operations for each layer with mapping
    for pruning_layer, stats in pruning_stats.items():
        if pruning_layer in layer_mapping:
            op_layer = layer_mapping[pruning_layer]
            
            if op_layer in base_operations:
                sparsity = stats['sparsity_percentage'] / 100.0  # Convert to fraction
                pruned_ops = base_operations[op_layer] * (1 - sparsity)
                pruned_operations[op_layer] = {
                    'base_ops': base_operations[op_layer],
                    'pruned_ops': pruned_ops,
                    'saved_ops': base_operations[op_layer] - pruned_ops,
                    'savings_percent': sparsity * 100
                }
                total_pruned_ops += pruned_ops
    
    # Add operations for any layers that didn't have pruning stats
    for op_layer, ops in base_operations.items():
        if op_layer != 'total' and op_layer not in pruned_operations:
            pruned_operations[op_layer] = {
                'base_ops': ops,
                'pruned_ops': ops,  # No pruning for this layer
                'saved_ops': 0,
                'savings_percent': 0
            }
            total_pruned_ops += ops
    
    # Calculate total operations
    total_base_ops = sum(layer_ops['base_ops'] for layer_ops in pruned_operations.values())
    pruned_operations['total'] = {
        'base_ops': total_base_ops,
        'pruned_ops': total_pruned_ops,
        'saved_ops': total_base_ops - total_pruned_ops,
        'savings_percent': (1 - total_pruned_ops / total_base_ops) * 100 if total_base_ops > 0 else 0
    }
    
    return pruned_operations

def format_ops(ops):
    """Format operations count to be more readable."""
    if ops >= 1e9:
        return f"{ops/1e9:.2f} G"
    elif ops >= 1e6:
        return f"{ops/1e6:.2f} M"
    elif ops >= 1e3:
        return f"{ops/1e3:.2f} K"
    else:
        return f"{ops:.2f}"
