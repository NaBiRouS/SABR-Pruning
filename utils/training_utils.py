"""
Training Utilities for SABR

This module contains utility functions for training, evaluating,
and analyzing models trained with SABR.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def evaluate_model(model, data_loader, criterion, device='cuda', use_gpu_dataset=False):
    """
    Evaluate a model on a given dataset.
    
    Args:
        model: The PyTorch model to evaluate
        data_loader: DataLoader for the dataset
        criterion: Loss function
        device: Device to evaluate on ('cuda' or 'cpu')
        use_gpu_dataset: Whether the dataset is already on GPU
        
    Returns:
        loss: Evaluation loss
        accuracy: Evaluation accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar for evaluation
    eval_pbar = tqdm(data_loader, desc="Evaluating")
    with torch.no_grad():
        for inputs, labels in eval_pbar:
            # Only transfer to device if not using GPU dataset
            if not use_gpu_dataset:
                inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            batch_loss = loss.item()
            batch_acc = predicted.eq(labels).sum().item() / labels.size(0)
            eval_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}'
            })
    
    loss = running_loss / len(data_loader.dataset)
    accuracy = correct / total
    
    return loss, accuracy


def create_summary_table(history_df, output_dir):
    """
    Create and save a summary table of training results.
    
    Args:
        history_df: DataFrame containing training history
        output_dir: Directory to save the table
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
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Table data
    table_data = [
        ['Metric', 'Final Value', 'Best Value', 'Best Epoch'],
        ['Training Loss', f"{final_metrics['Train Loss']:.6f}", f"{best_val_loss_metrics['Train Loss']:.6f}", f"{best_val_loss_metrics['Epoch']}" ],
        ['Validation Loss', f"{final_metrics['Val Loss']:.6f}", f"{best_val_loss_metrics['Val Loss']:.6f}", f"{best_val_loss_metrics['Epoch']}"],
        ['Training Accuracy', f"{final_metrics['Train Acc']:.2%}", f"{best_val_acc_metrics['Train Acc']:.2%}", f"{best_val_acc_metrics['Epoch']}"],
        ['Validation Accuracy', f"{final_metrics['Val Acc']:.2%}", f"{best_val_acc_metrics['Val Acc']:.2%}", f"{best_val_acc_metrics['Epoch']}"],
        ['Learning Rate', f"{final_metrics['Learning Rate']:.8f}", 'N/A', 'N/A'],
        ['Total Epochs', f"{final_epoch}", 'N/A', 'N/A'],
    ]
    
    # Add extra statistics
    max_gap = max([t - v for t, v in zip(history_df['Train Acc'], history_df['Val Acc'])])
    table_data.append(['Max Train-Val Acc Gap', f"{max_gap:.2%}", 'N/A', 'N/A'])
    
    # Create the table
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
    
    # Add title
    plt.title('Training Summary', fontsize=16, pad=20)
    
    # Save the table
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a text file summary
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write("TRAINING SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        for row in table_data:
            f.write(f"{row[0]}: {row[1]}")
            if row[2] != 'N/A':
                f.write(f" (Best: {row[2]} at epoch {row[3]})")
            f.write("\n")
            
        f.write("\n" + "="*50 + "\n")


def save_model(model, file_path, create_dir=True):
    """
    Save a PyTorch model to a file.
    
    Args:
        model: The PyTorch model to save
        file_path: Path to save the model to
        create_dir: Whether to create the directory if it doesn't exist
    """
    if create_dir:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def load_model(model, file_path, device='cuda'):
    """
    Load a PyTorch model from a file.
    
    Args:
        model: The PyTorch model to load into
        file_path: Path to load the model from
        device: Device to load the model to
        
    Returns:
        The loaded model
    """
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {file_path}")
    return model


def calculate_overall_sparsity(model, epsilon=1e-5):
    """
    Calculate the overall sparsity of a model.
    
    Args:
        model: The PyTorch model
        epsilon: Threshold for considering a weight as zero
        
    Returns:
        overall_sparsity: Overall sparsity percentage
        total_params: Total number of parameters
        zero_params: Number of parameters close to zero
    """
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider weight parameters
            # Get the weights as a numpy array
            weights = param.data.cpu().numpy()
            # Count zero and total weights
            total_params += weights.size
            zero_params += np.sum(np.abs(weights) < epsilon)
    
    overall_sparsity = (zero_params / total_params) * 100
    
    return overall_sparsity, total_params, zero_params
