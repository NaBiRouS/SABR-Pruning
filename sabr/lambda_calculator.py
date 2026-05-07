"""
Lambda Calculator for SABR

Provides functions to calculate layer-specific lambda values based on
weight distributions for the SABR algorithm.
"""

import numpy as np


def calculate_teta1_std_values(model, teta1):
    """
    Calculate teta1 * std(w) for each weight layer in the model.
    
    Args:
        model: The PyTorch model
        teta1: Multiplier for standard deviation
    
    Returns:
        Dictionary mapping layer names to teta1 * std values
    """
    teta1_std_values = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only process weight parameters
            # Convert to numpy and calculate std
            weights = param.data.cpu().numpy()
            std_value = np.std(weights)
            
            # Calculate teta1 * std
            teta1_std = teta1 * std_value
            
            # Store in dictionary
            teta1_std_values[name] = teta1_std
    
    return teta1_std_values


def calculate_filtered_std_values(model, teta1, gamma=0.1):
    """
    Calculate teta1 * std(w) for each weight layer, excluding weights below gamma * std.
    
    Args:
        model: The PyTorch model
        teta1: Multiplier for standard deviation
        gamma: Threshold for filtering weights (as a fraction of std)
    
    Returns:
        Dictionary mapping layer names to teta1 * filtered_std values
    """
    teta1_std_values = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only process weight parameters
            # Convert to numpy for calculation
            weights = param.data.cpu().numpy()
            
            # Calculate standard std
            std_value = np.std(weights)
            
            # Filter weights below gamma * std
            weights_for_std = weights[np.abs(weights) >= gamma * std_value]
            
            # If there are no weights above gamma, fallback to using all weights
            if len(weights_for_std) == 0:
                filtered_std = std_value
            else:
                filtered_std = np.std(weights_for_std)
            
            # Calculate teta1 * filtered_std
            teta1_std = teta1 * filtered_std
            
            # Store in dictionary
            teta1_std_values[name] = teta1_std
    
    return teta1_std_values
