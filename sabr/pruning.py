"""
Pruning Module for SABR

Provides functions for post-training pruning and sparsity analysis
for models trained with SABR.
"""

import torch
import numpy as np


def calculate_sparsity(model, epsilon=1e-5):
    """
    Calculate the sparsity (percentage of weights close to zero) for each layer.
    
    Args:
        model: The PyTorch model
        epsilon: Threshold for considering a weight as zero
        
    Returns:
        dict: Dictionary mapping layer names to their sparsity percentages
    """
    sparsity = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider weight parameters
            # Get the weights as a numpy array
            weights = param.data.cpu().numpy()
            # Calculate the percentage of weights with absolute value < epsilon
            zero_weights = np.sum(np.abs(weights) < epsilon)
            total_weights = weights.size
            sparsity_percentage = (zero_weights / total_weights) * 100
            sparsity[name] = sparsity_percentage
    
    return sparsity


def prune_model(model, epsilon=1e-5, use_std_based=False, teta1=0.5, gamma=0.1):
    """
    Prune the model by setting weights with absolute value less than threshold to zero.
    
    Args:
        model: The PyTorch model to prune
        epsilon: Threshold for pruning (used when use_std_based=False)
        use_std_based: If True, use teta1*std as the threshold, otherwise use epsilon
        teta1: Multiplier for standard deviation (used when use_std_based=True)
        gamma: Ignore weights with absolute values less than gamma * np.std(weights) when calculating std
        
    Returns:
        model: The pruned model
        pruning_stats: Dictionary containing pruning statistics for each layer
    """
    pruning_stats = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider weight parameters
            # Get original statistics
            weights = param.data.cpu().numpy()
            total_params = weights.size
            
            # Count non-zero weights before pruning
            non_zero_before = np.sum(np.abs(weights) > 0)
            
            # Calculate threshold based on the method
            if use_std_based:
                # Filter out weights with absolute values less than gamma * np.std(weights) for std calculation
                weights_for_std = weights[np.abs(weights) >= gamma * np.std(weights)]
                
                # If there are no weights above gamma, fallback to using all weights
                if len(weights_for_std) == 0:
                    std_value = np.std(weights)
                    filtered_count = 0
                else:
                    std_value = np.std(weights_for_std)
                    filtered_count = total_params - len(weights_for_std)
                
                threshold = teta1 * std_value
                print(f"Layer {name}: std = {std_value:.6f}, threshold = {threshold:.6f}")
                print(f"  (Ignored {filtered_count} weights < {gamma} when calculating std)")
            else:
                threshold = epsilon
            
            # Apply pruning
            with torch.no_grad():
                param.data[torch.abs(param.data) < threshold] = 0.0
            
            # Count non-zero weights after pruning
            weights_after = param.data.cpu().numpy()
            non_zero_after = np.sum(np.abs(weights_after) > 0)
            
            # Calculate sparsity percentage
            sparsity_percentage = ((total_params - non_zero_after) / total_params) * 100
             
            # Store statistics
            pruning_stats[name] = {
                'total_params': total_params,
                'non_zero_before': non_zero_before,
                'non_zero_after': non_zero_after,
                'pruned_params': total_params - non_zero_after,
                'sparsity_percentage': sparsity_percentage,
                'threshold': threshold
            }
            
            if use_std_based:
                pruning_stats[name]['std_value'] = std_value
                pruning_stats[name]['filtered_for_std'] = filtered_count
    
    return model, pruning_stats


def eliminate_dropout(model):
    """
    Set dropout rate to 0.0 for all dropout layers in a PyTorch model.
    Essential for post-training sparsity to ensure deterministic behavior.
    
    Args:
        model: A PyTorch model (nn.Module)
        
    Returns:
        The same model with all dropout probabilities set to 0.0
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d) or isinstance(module, torch.nn.Dropout3d):
            module.p = 0.0
            print(f"Set dropout probability to 0.0 for {module}")
    
    return model
