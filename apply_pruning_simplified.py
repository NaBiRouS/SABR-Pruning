"""
Simplified Pruning Function for Notebook

This module contains a simplified version of the apply_pruning function
for use in the notebook without computational savings calculations.
"""

import os
import torch
import torch.nn as nn

def apply_pruning_simplified(model, test_loader, output_dir='results/pruned_model', device='cuda',
                 use_gpu_dataset=False, use_std_based=True, teta1=0.2, gamma=0.1, epsilon=1e-5):
    """Apply post-training pruning to a model (simplified version for notebook)."""
    from sabr.pruning import prune_model, calculate_sparsity
    from utils.training_utils import evaluate_model, save_model, calculate_overall_sparsity
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Move model to device
    model.to(device)
    
    # Define loss function for evaluation
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate original model
    print("\nEvaluating original model...")
    original_loss, original_acc = evaluate_model(model, test_loader, criterion, device, use_gpu_dataset)
    print(f'Original Model - Test Loss: {original_loss:.4f}, Test Accuracy: {original_acc:.4f}')
    
    # Calculate initial sparsity
    initial_sparsity, total_params, initial_zero = calculate_overall_sparsity(model, epsilon)
    print(f"Initial sparsity: {initial_sparsity:.2f}% ({initial_zero}/{total_params} weights near zero)")
    
    # Apply pruning
    if use_std_based:
        print(f"\nApplying std-based pruning with teta1 = {teta1} and gamma = {gamma}...")
    else:
        print(f"\nApplying threshold-based pruning with epsilon = {epsilon}...")
    
    pruned_model, pruning_stats = prune_model(model, epsilon, use_std_based, teta1, gamma)
    
    # Evaluate pruned model
    print("\nEvaluating pruned model...")
    pruned_loss, pruned_acc = evaluate_model(pruned_model, test_loader, criterion, device, use_gpu_dataset)
    print(f'Pruned Model - Test Loss: {pruned_loss:.4f}, Test Accuracy: {pruned_acc:.4f}')
    print(f'Accuracy change: {pruned_acc - original_acc:.4f}')
    
    # Calculate overall sparsity
    total_params = sum(stats['total_params'] for stats in pruning_stats.values())
    total_non_zero = sum(stats['non_zero_after'] for stats in pruning_stats.values())
    total_pruned = total_params - total_non_zero
    overall_sparsity = (total_pruned / total_params) * 100
    
    # Print summary
    print("\n" + "="*50)
    print("PRUNING SUMMARY")
    print("="*50)
    if use_std_based:
        print(f"Pruning method: Standard deviation-based (teta1 = {teta1}, gamma = {gamma})")
    else:
        print(f"Pruning method: Threshold-based (epsilon = {epsilon})")
    print(f"Original accuracy: {original_acc:.4f}")
    print(f"Pruned accuracy: {pruned_acc:.4f}")
    print(f"Accuracy change: {pruned_acc - original_acc:.4f}")
    print(f"Initial sparsity: {initial_sparsity:.2f}%")
    print(f"Final sparsity: {overall_sparsity:.2f}%")
    print(f"Sparsity increase: {overall_sparsity - initial_sparsity:.2f}%")
    print(f"Total parameters: {total_params:,}")
    print(f"Non-zero parameters after pruning: {total_non_zero:,}")
    print(f"Pruned parameters: {total_pruned:,}")
    print("="*50)
    
    # Save the pruned model
    pruning_method = f"std_based_teta1_{teta1}_gamma_{gamma}" if use_std_based else f"threshold_based_epsilon_{epsilon}"
    model_filename = f'pruned_model_{pruning_method}.pth'
    save_model(pruned_model, os.path.join(output_dir, model_filename))
    
    # Create summary file
    with open(os.path.join(output_dir, 'pruning_summary.txt'), 'w') as f:
        f.write("PRUNING SUMMARY\n")
        f.write("="*50 + "\n")
        if use_std_based:
            f.write(f"Pruning method: Standard deviation-based (teta1 = {teta1}, gamma = {gamma})\n")
        else:
            f.write(f"Pruning method: Threshold-based (epsilon = {epsilon})\n")
        f.write(f"Original accuracy: {original_acc:.4f}\n")
        f.write(f"Pruned accuracy: {pruned_acc:.4f}\n")
        f.write(f"Accuracy change: {pruned_acc - original_acc:.4f}\n")
        f.write(f"Initial sparsity: {initial_sparsity:.2f}%\n")
        f.write(f"Final sparsity: {overall_sparsity:.2f}%\n")
        f.write(f"Sparsity increase: {overall_sparsity - initial_sparsity:.2f}%\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Non-zero parameters after pruning: {total_non_zero:,}\n")
        f.write(f"Pruned parameters: {total_pruned:,}\n")
        f.write("="*50 + "\n\n")
        
        f.write("LAYER-WISE SPARSITY\n")
        f.write("="*50 + "\n")
        for layer_name, stats in pruning_stats.items():
            f.write(f"Layer: {layer_name}\n")
            f.write(f"  Total Parameters: {stats['total_params']:,}\n")
            f.write(f"  Non-Zero Parameters: {stats['non_zero_after']:,}\n")
            f.write(f"  Pruned Parameters: {stats['pruned_params']:,}\n")
            f.write(f"  Sparsity: {stats['sparsity_percentage']:.2f}%\n")
            f.write(f"  Threshold: {stats['threshold']:.6f}\n")
            if use_std_based and 'std_value' in stats:
                f.write(f"  Standard Deviation: {stats['std_value']:.6f}\n")
            f.write("\n")
    
    # Return the pruned model, pruning stats, and results for further processing
    pruning_results = {
        'original_acc': original_acc,
        'pruned_acc': pruned_acc,
        'initial_sparsity': initial_sparsity,
        'final_sparsity': overall_sparsity,
        'total_params': total_params,
        'non_zero_params': total_non_zero,
        'pruned_params': total_pruned
    }
    
    return pruned_model, pruning_stats, pruning_results
