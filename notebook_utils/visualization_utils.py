"""
Visualization Utilities for SABR Notebooks

This module contains enhanced visualization functions for SABR (Sparsity-Accuracy Balanced 
Regularization) analysis and training results.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
import seaborn as sns

# Set Seaborn style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Define a better color palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def format_layer_name(layer_name):
    """
    Format layer name for better readability in plots.
    
    Args:
        layer_name: Original layer name from model
        
    Returns:
        Formatted layer name
    """
    # Replace dots with line breaks to improve readability
    parts = layer_name.split('.')
    
    # For VGG-style layers
    if len(parts) >= 2 and parts[0] in ['features', 'classifier']:
        if parts[0] == 'features':
            layer_type = "Conv" if parts[1] in ['0', '3', '6', '8', '11', '13', '16', '18'] else "Other"
            return f"{parts[0]}.{parts[1]}\n({layer_type})"
        else:
            return f"{parts[0]}.{parts[1]}\n(FC)"
            
    # For MobileNetV2-style layers
    if "conv" in layer_name or "block" in layer_name:
        # Shorten but keep descriptive
        return layer_name.replace("model.", "").replace("layers.", "")
        
    return layer_name


def plot_training_curves(epochs, train_losses, val_losses, train_accs, val_accs, lr_history, output_dir, 
                         title_prefix=""):
    """
    Enhanced plot of training curves showing loss, accuracy, learning rate, and train-val gap.
    
    Args:
        epochs: List of epoch numbers
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        lr_history: List of learning rates
        output_dir: Directory to save the plots
        title_prefix: Optional prefix for the plot title
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    if title_prefix:
        fig.suptitle(f'{title_prefix} Training and Validation Metrics', fontsize=16)
    else:
        fig.suptitle('Training and Validation Metrics', fontsize=16)
    
    # Plot loss curves with enhanced styling
    axs[0, 0].plot(epochs, train_losses, '-', color=COLORS[0], linewidth=2, 
                   marker='o', markersize=4, label='Training Loss')
    axs[0, 0].plot(epochs, val_losses, '-', color=COLORS[1], linewidth=2, 
                   marker='s', markersize=4, label='Validation Loss')
    axs[0, 0].set_title('Loss Curves', fontweight='bold')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend(frameon=True, fancybox=True, shadow=True)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy curves with enhanced styling
    axs[0, 1].plot(epochs, train_accs, '-', color=COLORS[0], linewidth=2, 
                   marker='o', markersize=4, label='Training Accuracy')
    axs[0, 1].plot(epochs, val_accs, '-', color=COLORS[1], linewidth=2, 
                   marker='s', markersize=4, label='Validation Accuracy')
    axs[0, 1].set_title('Accuracy Curves', fontweight='bold')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend(frameon=True, fancybox=True, shadow=True)
    axs[0, 1].grid(True, alpha=0.3)
    
    # Learning rate with enhanced styling (log scale)
    axs[1, 0].semilogy(epochs, lr_history, '-', color=COLORS[2], linewidth=2, marker='o', markersize=4)
    axs[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Learning Rate (log scale)')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Train-val accuracy gap with enhanced styling
    acc_gap = [t - v for t, v in zip(train_accs, val_accs)]
    axs[1, 1].plot(epochs, acc_gap, '-', color=COLORS[3], linewidth=2, marker='o', markersize=4)
    axs[1, 1].set_title('Train-Validation Accuracy Gap (Overfitting Indicator)', fontweight='bold')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Gap (Train Acc - Val Acc)')
    axs[1, 1].grid(True, alpha=0.3)
    
    # Add a horizontal line at y=0 for reference
    axs[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}training_curves.png'), dpi=300)
    plt.show()
    plt.close()


def plot_lambda_history(lambda_history, frozen_lambdas=None, output_dir='plots', 
                        title_prefix="", top_n_layers=5):
    """
    Enhanced plot of lambda value evolution with better layer naming and styling.
    
    Args:
        lambda_history: Dictionary mapping layer names to lists of lambda values
        frozen_lambdas: Set of layer names with frozen lambdas
        output_dir: Directory to save the plots
        title_prefix: Optional prefix for the plot title
        top_n_layers: Number of highest lambda layers to highlight individually
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for all layers
    plt.figure(figsize=(14, 10))
    
    # Generate epochs list (starting from 1)
    epochs = list(range(1, len(next(iter(lambda_history.values()))) + 1))
    
    # Get the layers with the highest final lambda values for individual plotting
    final_lambda_values = {layer: history[-1] for layer, history in lambda_history.items()}
    top_layers = sorted(final_lambda_values.items(), key=lambda x: x[1], reverse=True)[:top_n_layers]
    top_layer_names = [layer for layer, _ in top_layers]
    
    # Plot lambda values for all layers with log scale for y-axis
    for i, (layer_name, history) in enumerate(lambda_history.items()):
        color_idx = i % len(COLORS)
        if layer_name in top_layer_names:
            # Use layer index in top layers to determine color
            top_idx = top_layer_names.index(layer_name)
            # Highlight top layers with thicker lines and markers
            plt.semilogy(epochs, history, linewidth=2.5, 
                       marker='o', markersize=5, markevery=2,
                       color=COLORS[top_idx],
                       label=format_layer_name(layer_name))
        else:
            # Plot other layers with thinner, semi-transparent lines
            plt.semilogy(epochs, history, linewidth=0.8, alpha=0.2, 
                       color=COLORS[color_idx % len(COLORS)])
    
    # Mark frozen lambdas with vertical lines and annotations
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
                if frozen_epoch < len(epochs) and layer_name in top_layer_names:
                    plt.axvline(x=epochs[frozen_epoch], color=COLORS[top_layer_names.index(layer_name)], 
                               linestyle='--', alpha=0.5)
                    # Add annotation for top layers only
                    if layer_name in top_layer_names:
                        plt.annotate(f"Frozen", 
                                   xy=(epochs[frozen_epoch], lambda_history[layer_name][frozen_epoch]),
                                   xytext=(5, 0), textcoords='offset points',
                                   fontsize=8, rotation=90, alpha=0.7,
                                   color=COLORS[top_layer_names.index(layer_name)])
    
    # Add labels and title
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Lambda Value (log scale)', fontsize=12, fontweight='bold')
    if title_prefix:
        plt.title(f'{title_prefix} Layer-specific L1 Regularization Lambda Evolution', 
                 fontsize=14, fontweight='bold')
    else:
        plt.title('Layer-specific L1 Regularization Lambda Evolution', 
                 fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add annotations about the algorithm behavior
    plt.annotate("↑ Lambda increases when accuracy is stable", 
               xy=(0.5, 0.02), xycoords='figure fraction',
               fontsize=10, ha='center', 
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
    
    plt.annotate("↓ Lambda decreases when accuracy drops", 
               xy=(0.5, 0.06), xycoords='figure fraction',
               fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}lambda_history_all_layers.png'), dpi=300)
    plt.show()
    plt.close()
    
    # Create a second plot showing average lambda value across all layers
    plt.figure(figsize=(12, 6))
    
    # Calculate average lambda value per epoch
    avg_lambda_history = []
    for epoch_idx in range(len(epochs)):
        epoch_avg = np.mean([history[epoch_idx] for history in lambda_history.values()])
        avg_lambda_history.append(epoch_avg)
    
    # Plot average lambda values with enhanced styling
    plt.semilogy(epochs, avg_lambda_history, color=COLORS[0], linewidth=2.5, 
               marker='o', markersize=6, markevery=2)
    
    # Add labels and title
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Average Lambda Value (log scale)', fontsize=12, fontweight='bold')
    if title_prefix:
        plt.title(f'{title_prefix} Average L1 Regularization Lambda Evolution', 
                 fontsize=14, fontweight='bold')
    else:
        plt.title('Average L1 Regularization Lambda Evolution', 
                 fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key points with improved styling
    plt.annotate(f"Start: {avg_lambda_history[0]:.8f}", 
                xy=(epochs[0], avg_lambda_history[0]), 
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='orange', alpha=0.8))
    
    plt.annotate(f"End: {avg_lambda_history[-1]:.8f}", 
                xy=(epochs[-1], avg_lambda_history[-1]), 
                xytext=(-80, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='orange', alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}lambda_history_average.png'), dpi=300)
    plt.show()
    plt.close()


def plot_sparsity(sparsity_dict, epoch, output_dir='plots', title_prefix=""):
    """
    Enhanced plot of sparsity statistics for each layer with better formatting.
    
    Args:
        sparsity_dict: Dictionary mapping layer names to their sparsity percentages
        epoch: Current epoch number
        output_dir: Directory to save the plots
        title_prefix: Optional prefix for the plot title
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract layer names and sparsity values
    layer_names = list(sparsity_dict.keys())
    formatted_names = [format_layer_name(name) for name in layer_names]
    sparsity_values = [sparsity_dict[name] for name in layer_names]
    
    # Sort by sparsity values for better visualization
    sorted_indices = np.argsort(sparsity_values)[::-1]  # Descending order
    sorted_names = [formatted_names[i] for i in sorted_indices]
    sorted_values = [sparsity_values[i] for i in sorted_indices]
    original_names = [layer_names[i] for i in sorted_indices]
    
    # Create the plot with enhanced styling
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(sorted_names)), sorted_values, width=0.7, 
                  color=[to_rgba(COLORS[0], 0.7 + 0.3 * (val/100)) for val in sorted_values])
    
    # Add labels and title with improved styling
    plt.xlabel('Layers', fontsize=12, fontweight='bold')
    plt.ylabel('Sparsity Percentage (%)', fontsize=12, fontweight='bold')
    if title_prefix:
        plt.title(f'{title_prefix} Layer Sparsity (Epoch {epoch})', 
                 fontsize=14, fontweight='bold')
    else:
        plt.title(f'Layer Sparsity (Epoch {epoch})', 
                 fontsize=14, fontweight='bold')
    
    # Customize x-axis ticks to show layer names
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    
    # Add values on top of the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{sorted_values[i]:.1f}%', 
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.7))
    
    # Add a horizontal grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Add an annotation with the overall average sparsity
    avg_sparsity = np.mean(sorted_values)
    plt.axhline(y=avg_sparsity, color='red', linestyle='--', alpha=0.7)
    plt.annotate(f'Average: {avg_sparsity:.1f}%', 
               xy=(len(sorted_names)-1, avg_sparsity), 
               xytext=(0, 10), textcoords='offset points',
               fontsize=10, color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}sparsity_epoch_{epoch}.png'), dpi=300)
    plt.show()
    plt.close()
    
    # Return sorted data for possible further use
    return {
        'layer_names': original_names,
        'formatted_names': sorted_names,
        'sparsity_values': sorted_values
    }


def plot_weight_distributions(model, output_dir='plots', title_prefix="", epsilon=1e-5):
    """
    Enhanced plot of weight distributions for layers in the model with clear layer naming.
    
    Args:
        model: The trained PyTorch model
        output_dir: Directory to save the plots
        title_prefix: Optional prefix for the plot title
        epsilon: Threshold for considering weights as zero
    """
    # Create a directory for weight distribution plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data for all layers
    layer_data = []
    
    # Iterate through named parameters of the model
    for name, param in model.named_parameters():
        # Only plot weight parameters (skip biases)
        if 'weight' in name:
            # Convert weights to numpy for plotting
            weights = param.data.cpu().numpy()
            
            # Flatten the weights to 1D array
            weights_flat = weights.flatten()
            
            # Calculate statistics
            mean = np.mean(weights_flat)
            median = np.median(weights_flat)
            std = np.std(weights_flat)
            zero_count = np.sum(np.abs(weights_flat) < epsilon)
            zero_percent = (zero_count / len(weights_flat)) * 100
            
            # Store data
            layer_data.append({
                'name': name,
                'formatted_name': format_layer_name(name),
                'weights': weights_flat,
                'mean': mean,
                'median': median,
                'std': std,
                'min': np.min(weights_flat),
                'max': np.max(weights_flat),
                'zero_percent': zero_percent
            })
    
    # Sort layers by sparsity (zero_percent) for better presentation
    layer_data.sort(key=lambda x: x['zero_percent'], reverse=True)
    
    # Plot distributions for top N sparse layers
    top_n = min(10, len(layer_data))
    
    # Create a multi-panel figure for top sparse layers
    fig, axs = plt.subplots(top_n, 1, figsize=(12, 3*top_n), sharex=True)
    
    for i in range(top_n):
        layer = layer_data[i]
        ax = axs[i] if top_n > 1 else axs
        
        # Plot histogram with KDE
        sns.histplot(layer['weights'], bins=50, kde=True, ax=ax, color=COLORS[i % len(COLORS)])
        
        # Add vertical lines for important statistics
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=layer['mean'], color='g', linestyle='-', alpha=0.7, 
                  label=f"Mean: {layer['mean']:.4f}")
        ax.axvline(x=layer['median'], color='y', linestyle='-', alpha=0.7,
                  label=f"Median: {layer['median']:.4f}")
        
        # Highlight epsilon threshold regions
        ax.axvspan(-epsilon, epsilon, alpha=0.2, color='red', 
                  label=f"<{epsilon:.5f}: {layer['zero_percent']:.1f}%")
        
        # Add layer name and statistics
        title = f"{layer['formatted_name']} - Sparsity: {layer['zero_percent']:.1f}%, Std: {layer['std']:.4f}"
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        
        # Only show x label on bottom plot
        if i == top_n-1:
            ax.set_xlabel('Weight Value')
        
        ax.set_ylabel('Frequency')
    
    # Add overall title
    if title_prefix:
        fig.suptitle(f'{title_prefix} Weight Distributions for Top {top_n} Sparse Layers', 
                    fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Weight Distributions for Top {top_n} Sparse Layers', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}weight_distributions_top_sparse.png'), dpi=300)
    plt.show()
    plt.close()
    
    # Create a sparsity summary bar chart
    plt.figure(figsize=(12, 6))
    
    # Extract data for the bar chart
    names = [layer['formatted_name'] for layer in layer_data]
    sparsity_values = [layer['zero_percent'] for layer in layer_data]
    
    # Create bars with a color gradient based on sparsity
    bars = plt.bar(range(len(names)), sparsity_values, width=0.7,
                  color=[to_rgba(COLORS[0], 0.5 + 0.5 * (val/100)) for val in sparsity_values])
    
    # Add labels and title
    plt.xlabel('Layers', fontsize=12, fontweight='bold')
    plt.ylabel('Sparsity (%)', fontsize=12, fontweight='bold')
    if title_prefix:
        plt.title(f'{title_prefix} Weight Sparsity by Layer', 
                 fontsize=14, fontweight='bold')
    else:
        plt.title('Weight Sparsity by Layer', fontsize=14, fontweight='bold')
    
    # Customize x-axis ticks
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    
    # Add values on top of the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{sparsity_values[i]:.1f}%', ha='center', va='bottom')
    
    # Add a horizontal grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Add overall average sparsity
    avg_sparsity = np.mean(sparsity_values)
    plt.axhline(y=avg_sparsity, color='red', linestyle='--', alpha=0.7)
    plt.annotate(f'Average: {avg_sparsity:.1f}%', 
               xy=(len(names)-1, avg_sparsity), 
               xytext=(0, 10), textcoords='offset points',
               fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}weight_sparsity_summary.png'), dpi=300)
    plt.show()
    plt.close()


def plot_sparsity_evolution(history_df, output_dir='plots', title_prefix=""):
    """
    Plot the evolution of model sparsity over epochs.
    
    Args:
        history_df: DataFrame with epoch and sparsity information
        output_dir: Directory to save the plot
        title_prefix: Optional prefix for the plot title
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    
    # Plot sparsity evolution with enhanced styling
    plt.plot(history_df['Epoch'], history_df['Sparsity'], '-', color=COLORS[0], linewidth=2.5, 
           marker='o', markersize=6, markevery=max(1, len(history_df) // 20))
    
    # Set labels and title with improved styling
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Sparsity (%)', fontsize=12, fontweight='bold')
    if title_prefix:
        plt.title(f'{title_prefix} Overall Model Sparsity Evolution During SABR Training', 
                 fontsize=14, fontweight='bold')
    else:
        plt.title('Overall Model Sparsity Evolution During SABR Training', 
                 fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Add annotations for initial and final sparsity values
    plt.annotate(f"Initial: {history_df['Sparsity'].iloc[0]:.2f}%", 
                xy=(history_df['Epoch'].iloc[0], history_df['Sparsity'].iloc[0]),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, arrowprops=dict(arrowstyle='->'))
    
    plt.annotate(f"Final: {history_df['Sparsity'].iloc[-1]:.2f}%", 
                xy=(history_df['Epoch'].iloc[-1], history_df['Sparsity'].iloc[-1]),
                xytext=(-70, 20), textcoords='offset points',
                fontsize=10, arrowprops=dict(arrowstyle='->'))
    
    # Add a line showing the trend
    if len(history_df) > 1:
        z = np.polyfit(history_df['Epoch'], history_df['Sparsity'], 1)
        p = np.poly1d(z)
        plt.plot(history_df['Epoch'], p(history_df['Epoch']), "--", color='gray', alpha=0.5,
               label=f"Trend: {z[0]:.2f}% per epoch")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}sparsity_evolution.png'), dpi=300)
    plt.show()
    plt.close()


def create_summary_table(history_df, output_dir, title_prefix=""):
    """
    Create and save an enhanced summary table of training results.
    
    Args:
        history_df: DataFrame containing training history
        output_dir: Directory to save the table
        title_prefix: Optional prefix for the plot title
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get final and best metrics
    final_epoch = history_df['Epoch'].max()
    final_metrics = history_df[history_df['Epoch'] == final_epoch].iloc[0]
    
    best_val_acc_idx = history_df['Val Acc'].idxmax()
    best_val_acc_metrics = history_df.iloc[best_val_acc_idx]
    
    best_val_loss_idx = history_df['Val Loss'].idxmin()
    best_val_loss_metrics = history_df.iloc[best_val_loss_idx]
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('tight')
    ax.axis('off')
    
    # Table data
    table_data = [
        ['Metric', 'Final Value', 'Best Value', 'Best Epoch'],
        ['Training Loss', f"{final_metrics['Train Loss']:.6f}", 
         f"{best_val_loss_metrics['Train Loss']:.6f}", f"{best_val_loss_metrics['Epoch']}"],
        ['Validation Loss', f"{final_metrics['Val Loss']:.6f}", 
         f"{best_val_loss_metrics['Val Loss']:.6f}", f"{best_val_loss_metrics['Epoch']}"],
        ['Training Accuracy', f"{final_metrics['Train Acc']:.2%}", 
         f"{best_val_acc_metrics['Train Acc']:.2%}", f"{best_val_acc_metrics['Epoch']}"],
        ['Validation Accuracy', f"{final_metrics['Val Acc']:.2%}", 
         f"{best_val_acc_metrics['Val Acc']:.2%}", f"{best_val_acc_metrics['Epoch']}"],
        ['Learning Rate', f"{final_metrics['Learning Rate']:.8f}", 'N/A', 'N/A'],
        ['Total Epochs', f"{final_epoch}", 'N/A', 'N/A'],
    ]
    
    # Add sparsity statistics if available
    if 'Sparsity' in history_df.columns:
        sparsity_data = [
            ['Model Sparsity', f"{final_metrics['Sparsity']:.2f}%", 
             f"{history_df['Sparsity'].max():.2f}%", f"{history_df['Sparsity'].idxmax() + 1}"]
        ]
        table_data.extend(sparsity_data)
    
    # Add extra statistics
    max_gap = max([t - v for t, v in zip(history_df['Train Acc'], history_df['Val Acc'])])
    table_data.append(['Max Train-Val Acc Gap', f"{max_gap:.2%}", 'N/A', 'N/A'])
    
    # Create the table with improved styling
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Style header row
    for j, cell in enumerate(table._cells[(0, j)] for j in range(4)):
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    
    # Style data rows
    for i in range(1, len(table_data)):
        cell = table._cells[(i, 0)]
        cell.set_text_props(fontweight='bold')
        
        # Highlight best values column
        best_cell = table._cells[(i, 2)]
        best_cell.set_facecolor('#E6F0FF')
    
    # Add title with optional prefix
    if title_prefix:
        plt.title(f'{title_prefix} Training Summary', fontsize=16, pad=20, fontweight='bold')
    else:
        plt.title('Training Summary', fontsize=16, pad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}training_summary.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_computational_savings(pruned_operations, pruning_results, output_dir='plots', title_prefix=""):
    """
    Create enhanced visualizations for computational savings from pruning.
    
    Args:
        pruned_operations: Dictionary with operations information
        pruning_results: Dictionary with pruning statistics
        output_dir: Directory to save the visualizations
        title_prefix: Optional prefix for the plot title
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Format operations count to be more readable
    def format_ops(ops):
        if ops >= 1e9:
            return f"{ops/1e9:.2f} G"
        elif ops >= 1e6:
            return f"{ops/1e6:.2f} M"
        elif ops >= 1e3:
            return f"{ops/1e3:.2f} K"
        else:
            return f"{ops:.2f}"
    
    # Create a bar chart comparing base vs pruned operations
    plt.figure(figsize=(14, 8))
    
    # Extract data for non-total layers
    layer_names = []
    formatted_names = []
    base_ops = []
    pruned_ops = []
    savings_percent = []
    
    for layer_name, layer_ops in pruned_operations.items():
        if layer_name != 'total':
            layer_names.append(layer_name)
            formatted_names.append(format_layer_name(layer_name))
            base_ops.append(layer_ops['base_ops'])
            pruned_ops.append(layer_ops['pruned_ops'])
            savings_percent.append(layer_ops['savings_percent'])
    
    # Sort by savings percentage for better visualization
    sorted_indices = np.argsort(savings_percent)[::-1]  # Descending order
    layer_names = [layer_names[i] for i in sorted_indices]
    formatted_names = [formatted_names[i] for i in sorted_indices]
    base_ops = [base_ops[i] for i in sorted_indices]
    pruned_ops = [pruned_ops[i] for i in sorted_indices]
    savings_percent = [savings_percent[i] for i in sorted_indices]
    
    # Bar chart width
    bar_width = 0.35
    index = np.arange(len(layer_names))
    
    # Create grouped bars
    plt.bar(index, base_ops, bar_width, color=COLORS[0], alpha=0.8, label='Base Operations')
    plt.bar(index + bar_width, pruned_ops, bar_width, color=COLORS[1], alpha=0.8, label='Pruned Operations')
    
    # Add savings percentage as text
    for i, (base, pruned, saving) in enumerate(zip(base_ops, pruned_ops, savings_percent)):
        plt.text(i + bar_width/2, max(base, pruned) * 1.05, 
                f"{saving:.1f}% saved", ha='center', va='bottom', 
                rotation=45, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7))
    
    # Add labels and title with improved styling
    plt.xlabel('Layers', fontsize=12, fontweight='bold')
    plt.ylabel('Operations (MACs)', fontsize=12, fontweight='bold')
    if title_prefix:
        plt.title(f'{title_prefix} Computational Savings by Layer', 
                 fontsize=14, fontweight='bold')
    else:
        plt.title('Computational Savings by Layer', fontsize=14, fontweight='bold')
    
    plt.xticks(index + bar_width / 2, formatted_names, rotation=45, ha='right')
    plt.yscale('log')  # Log scale for better visualization of different magnitudes
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}computational_savings_by_layer.png'), dpi=300)
    plt.show()
    plt.close()
    
    # Create a pie chart for overall savings
    plt.figure(figsize=(10, 8))
    
    # Data for the pie chart
    labels = ['Remaining Operations', 'Saved Operations']
    sizes = [100 - pruned_operations['total']['savings_percent'], pruned_operations['total']['savings_percent']]
    colors = [COLORS[1], COLORS[0]]
    explode = (0, 0.1)  # Explode the saved slice
    
    # Create an enhanced pie chart
    patches, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', shadow=True, startangle=90,
                                      textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    
    # Add title and subtitle
    if title_prefix:
        plt.title(f'{title_prefix} Overall Computational Savings', 
                 fontsize=14, fontweight='bold')
    else:
        plt.title('Overall Computational Savings', fontsize=14, fontweight='bold')
    
    plt.annotate(f"Base: {format_ops(pruned_operations['total']['base_ops'])} MACs\n"
                f"Pruned: {format_ops(pruned_operations['total']['pruned_ops'])} MACs\n"
                f"Saved: {format_ops(pruned_operations['total']['saved_ops'])} MACs", 
                xy=(0, -0.1), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}computational_savings_overall.png'), dpi=300)
    plt.show()
    plt.close()
    
    # Create a scatter plot of sparsity vs computational savings
    plt.figure(figsize=(10, 8))
    
    # Extract data for the scatter plot
    scatter_layer_names = []
    sparsity_values = []
    op_savings = []
    op_counts = []
    
    # Get layer sparsity information
    for layer_name, layer_ops in pruned_operations.items():
        if layer_name != 'total':
            scatter_layer_names.append(format_layer_name(layer_name))
            sparsity_values.append(layer_ops['savings_percent'])  # Using savings as sparsity
            op_savings.append(layer_ops['saved_ops'])
            op_counts.append(layer_ops['base_ops'])
    
    # Create scatter plot with size proportional to operations count
    max_size = 1000  # Maximum marker size
    normalized_sizes = [max_size * count / max(op_counts) for count in op_counts]
    
    plt.scatter(sparsity_values, op_savings, s=normalized_sizes, 
              c=range(len(scatter_layer_names)), cmap='viridis', alpha=0.7)
    
    # Add layer labels to points
    for i, txt in enumerate(scatter_layer_names):
        plt.annotate(txt, (sparsity_values[i], op_savings[i]),
                   fontsize=8, ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7))
    
    # Add labels and title
    plt.xlabel('Sparsity / Savings Percentage (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Operations Saved (MACs)', fontsize=12, fontweight='bold')
    if title_prefix:
        plt.title(f'{title_prefix} Sparsity vs. Computational Savings', 
                 fontsize=14, fontweight='bold')
    else:
        plt.title('Sparsity vs. Computational Savings', fontsize=14, fontweight='bold')
    
    plt.colorbar(label='Layer Index')
    plt.grid(alpha=0.3)
    
    # Add a logarithmic scale for operations
    plt.yscale('log')
    
    # Add a reference line for perfect correlation
    max_x = max(sparsity_values)
    max_y = max(op_savings)
    plt.plot([0, max_x], [0, max_y * (max_x/100)], 'r--', alpha=0.5, 
            label='Perfect Correlation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}sparsity_vs_savings.png'), dpi=300)
    plt.show()
    plt.close()


def create_final_summary_table(base_val_acc, sabr_val_acc, pruning_results, pruned_operations, output_dir):
    """
    Create an enhanced final summary table comparing all phases.
    
    Args:
        base_val_acc: Validation accuracy of the base model
        sabr_val_acc: Validation accuracy of the SABR model
        pruning_results: Dictionary with pruning statistics
        pruned_operations: Dictionary with operations information
        output_dir: Directory to save the table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary table
    summary_data = [
        ["Phase", "Model", "Validation Acc", "Test Acc", "Sparsity", "Ops Reduction"],
        ["Phase 1", "Base Model", f"{base_val_acc:.4f}", "-", "0.00%", "0.00%"],
        ["Phase 2", "SABR Model", f"{sabr_val_acc:.4f}", "-", 
         f"{pruning_results['initial_sparsity']:.2f}%", "-"],
        ["Phase 3", "Pruned Model", "-", f"{pruning_results['pruned_acc']:.4f}", 
         f"{pruning_results['final_sparsity']:.2f}%", 
         f"{pruned_operations['total']['savings_percent']:.2f}%"]
    ]
    
    # Create a figure for the table with improved styling
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Style header row
    for j, cell in enumerate(table._cells[(0, j)] for j in range(6)):
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    
    # Style each phase row with different colors
    colors = ['#E6F0FF', '#E6F5E6', '#FFF0E6']
    for i in range(1, 4):
        for j in range(6):
            cell = table._cells[(i, j)]
            cell.set_facecolor(colors[i-1])
    
    plt.title('SABR Training Pipeline Summary', fontsize=16, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Create a horizontal bar chart showing accuracy vs sparsity tradeoff
    plt.figure(figsize=(12, 6))
    
    # Data for the bar chart
    phases = ['Base Model', 'SABR Model', 'Pruned Model']
    accuracy_values = [base_val_acc, sabr_val_acc, pruning_results['pruned_acc']]
    sparsity_values = [0, pruning_results['initial_sparsity'], pruning_results['final_sparsity']]
    
    # Create a twinx plot with bars for accuracy and line for sparsity
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot accuracy bars
    bars = ax1.bar(phases, accuracy_values, width=0.6, color=[COLORS[i] for i in [0, 1, 2]], alpha=0.7)
    
    # Add accuracy values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.001, 
                f"{accuracy_values[i]:.4f}", ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # Plot sparsity line
    ax2.plot(phases, sparsity_values, 'r-o', linewidth=2, markersize=8, label='Sparsity')
    
    # Add sparsity values near line points
    for i, (x, y) in enumerate(zip(phases, sparsity_values)):
        ax2.text(i, y + 2, f"{y:.2f}%", ha='center', va='bottom',
                fontsize=10, color='red', fontweight='bold')
    
    # Set labels and title
    ax1.set_xlabel('Pipeline Phase', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sparsity (%)', fontsize=12, fontweight='bold', color='red')
    plt.title('Accuracy vs. Sparsity Tradeoff', fontsize=14, fontweight='bold')
    
    # Set ticks and grid
    ax1.set_ylim(0, 1.05 * max(accuracy_values))
    ax2.set_ylim(0, 1.1 * max(sparsity_values))
    ax2.tick_params(axis='y', colors='red')
    ax1.grid(False)
    ax2.grid(False)
    
    # Add a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, ['Accuracy'] + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_sparsity.png'), dpi=300)
    plt.show()
    plt.close()
    
    # Save the final summary to a file
    with open(os.path.join(output_dir, 'final_summary.txt'), 'w') as f:
        f.write("SABR TRAINING PIPELINE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Phase 1: Base Model Training\n")
        f.write(f"  Validation Accuracy: {base_val_acc:.4f}\n")
        f.write(f"  Sparsity: 0.00%\n\n")
        
        f.write("Phase 2: SABR Regularization\n")
        f.write(f"  Validation Accuracy: {sabr_val_acc:.4f}\n")
        f.write(f"  Sparsity: {pruning_results['initial_sparsity']:.2f}%\n\n")
        
        f.write("Phase 3: Post-Training Pruning\n")
        f.write(f"  Test Accuracy: {pruning_results['pruned_acc']:.4f}\n")
        f.write(f"  Final Sparsity: {pruning_results['final_sparsity']:.2f}%\n")
        f.write(f"  Operations Reduction: {pruned_operations['total']['savings_percent']:.2f}%\n")
        f.write(f"  Total Parameters: {pruning_results['total_params']:,}\n")
        f.write(f"  Non-zero Parameters: {pruning_results['non_zero_params']:,}\n")
        f.write(f"  Zero Parameters: {pruning_results['pruned_params']:,}\n\n")
        
        f.write("Conclusion\n")
        f.write("----------\n")
        accuracy_change = pruning_results['pruned_acc'] - pruning_results['original_acc']
        f.write(f"The SABR pruning approach successfully reduced computational requirements ")
        f.write(f"by {pruned_operations['total']['savings_percent']:.2f}% ")
        f.write(f"while maintaining model accuracy (change of {accuracy_change:.4f})\n")
    
    return os.path.join(output_dir, 'final_summary.txt')
