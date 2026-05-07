"""
Visualization Utilities for SABR

This module contains functions for visualizing training progress,
model sparsity, and lambda values during SABR training.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(epochs, train_losses, val_losses, train_accs, val_accs, lr_history, output_dir):
    """
    Plot training curves showing loss, accuracy, learning rate, and train-val gap.
    
    Args:
        epochs: List of epoch numbers
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        lr_history: List of learning rates
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training and Validation Metrics', fontsize=16)
    
    # Plot loss curves
    axs[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axs[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axs[0, 0].set_title('Loss Curves')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot accuracy curves
    axs[0, 1].plot(epochs, train_accs, 'b-', label='Training Accuracy')
    axs[0, 1].plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    axs[0, 1].set_title('Accuracy Curves')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot learning rate
    axs[1, 0].semilogy(epochs, lr_history, 'g-')
    axs[1, 0].set_title('Learning Rate Schedule')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Learning Rate (log scale)')
    axs[1, 0].grid(True)
    
    # Plot train-val accuracy gap
    acc_gap = [t - v for t, v in zip(train_accs, val_accs)]
    axs[1, 1].plot(epochs, acc_gap, 'm-')
    axs[1, 1].set_title('Train-Validation Accuracy Gap (Overfitting Indicator)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Gap (Train Acc - Val Acc)')
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()


def plot_lambda_history(lambda_history, frozen_lambdas=None, output_dir='plots', top_n_layers=5):
    """
    Plot the history of lambda values across epochs.
    
    Args:
        lambda_history: Dictionary mapping layer names to lists of lambda values
        frozen_lambdas: Set of layer names with frozen lambdas
        output_dir: Directory to save the plots
        top_n_layers: Number of highest lambda layers to highlight individually
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for all layers
    plt.figure(figsize=(12, 8))
    
    # Generate epochs list (starting from 1)
    epochs = list(range(1, len(next(iter(lambda_history.values()))) + 1))
    
    # Get the layers with the highest final lambda values for individual plotting
    final_lambda_values = {layer: history[-1] for layer, history in lambda_history.items()}
    top_layers = sorted(final_lambda_values.items(), key=lambda x: x[1], reverse=True)[:top_n_layers]
    top_layer_names = [layer for layer, _ in top_layers]
    
    # Plot lambda values for all layers with log scale for y-axis
    for layer_name, history in lambda_history.items():
        if layer_name in top_layer_names:
            # Highlight top layers with thicker lines and markers
            plt.semilogy(epochs, history, linewidth=2, marker='o', label=f"{layer_name.split('.')[-2]}")
        else:
            # Plot other layers with thinner, semi-transparent lines
            plt.semilogy(epochs, history, linewidth=1, alpha=0.3)
    
    # Mark frozen lambdas with vertical lines
    if frozen_lambdas:
        for layer_name in frozen_lambdas:
            if layer_name in lambda_history:
                # Find the epoch where lambda was frozen (last value change)
                frozen_epoch = len(lambda_history[layer_name])
                for i in range(1, len(lambda_history[layer_name])):
                    if lambda_history[layer_name][i] == lambda_history[layer_name][i-1]:
                        frozen_epoch = i
                        break
                
                # Only add vertical line if we can identify when it was frozen
                if frozen_epoch < len(epochs):
                    plt.axvline(x=epochs[frozen_epoch], color='r', linestyle='--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Lambda Value (log scale)', fontsize=12)
    plt.title('Layer-specific L1 Regularization Lambda Evolution', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lambda_history_all_layers.png'), dpi=300)
    plt.close()
    
    # Create a second plot showing average lambda value across all layers
    plt.figure(figsize=(10, 6))
    
    # Calculate average lambda value per epoch
    avg_lambda_history = []
    for epoch_idx in range(len(epochs)):
        epoch_avg = np.mean([history[epoch_idx] for history in lambda_history.values()])
        avg_lambda_history.append(epoch_avg)
    
    # Plot average lambda values
    plt.semilogy(epochs, avg_lambda_history, 'r-o', linewidth=2, markersize=6)
    
    # Add labels and title
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Lambda Value (log scale)', fontsize=12)
    plt.title('Average L1 Regularization Lambda Evolution', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key points
    plt.annotate(f"Start: {avg_lambda_history[0]:.8f}", 
                xy=(1, avg_lambda_history[0]), 
                xytext=(5, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    
    plt.annotate(f"End: {avg_lambda_history[-1]:.8f}", 
                xy=(len(epochs), avg_lambda_history[-1]), 
                xytext=(-70, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lambda_history_average.png'), dpi=300)
    plt.close()


def plot_sparsity(sparsity_dict, epoch, output_dir='plots'):
    """
    Plot and save the sparsity statistics for each layer.
    
    Args:
        sparsity_dict: Dictionary mapping layer names to their sparsity percentages
        epoch: Current epoch number
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract layer names and sparsity values
    layer_names = list(sparsity_dict.keys())
    sparsity_values = [sparsity_dict[name] for name in layer_names]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(layer_names)), sparsity_values, width=0.7)
    
    # Add labels and title
    plt.xlabel('Layers')
    plt.ylabel('Sparsity Percentage (%)')
    plt.title(f'Layer Sparsity (Epoch {epoch})')
    
    # Customize x-axis ticks to show layer names
    plt.xticks(range(len(layer_names)), [name.replace('.', '\n') for name in layer_names], rotation=45, ha='right')
    
    # Add values on top of the bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{sparsity_values[i]:.2f}%', ha='center')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'sparsity_epoch_{epoch}.png'))
    plt.close()
    
    
def plot_weight_distributions(model, output_dir='plots'):
    """
    Plot histograms of weight distributions for each layer of the model.
    
    Args:
        model: The trained PyTorch model
        output_dir: Directory to save the plots
    """
    # Create a directory for weight distribution plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Counter for naming the plots
    layer_count = 0
    
    # Iterate through named parameters of the model
    for name, param in model.named_parameters():
        # Only plot weight parameters (skip biases)
        if 'weight' in name:
            # Convert weights to numpy for plotting
            weights = param.data.cpu().numpy()
            
            # Flatten the weights to 1D array
            weights_flat = weights.flatten()
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.hist(weights_flat, bins=50, alpha=0.7)
            plt.title(f'Weight Distribution: {name}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Add statistical information
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            plt.axvline(x=np.mean(weights_flat), color='g', linestyle='-', alpha=0.5, 
                       label=f'Mean: {np.mean(weights_flat):.4f}')
            plt.axvline(x=np.median(weights_flat), color='y', linestyle='-', alpha=0.5,
                       label=f'Median: {np.median(weights_flat):.4f}')
            
            # Add standard deviation information
            plt.text(0.02, 0.95, f'Std Dev: {np.std(weights_flat):.4f}', 
                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
            
            # Add L1 norm information (relevant for L1 regularization)
            l1_norm = np.sum(np.abs(weights_flat))
            plt.text(0.02, 0.90, f'L1 Norm: {l1_norm:.4f}', 
                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
            
            plt.legend()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f'layer_{layer_count}_{name.replace(".", "_")}.png'))
            plt.close()
            
            layer_count += 1
