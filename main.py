"""
SABR (Sparsity-Accuracy Balanced Regularization) Example

This script demonstrates a complete workflow for training a model with SABR:
1. Train a base model with standard techniques
2. Apply SABR regularization to induce sparsity
3. Perform post-training pruning for deployment
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import pandas as pd
from tqdm import tqdm

# Import SABR components
from sabr.sabr import SABR
from sabr.pruning import prune_model, eliminate_dropout, calculate_sparsity

# Import utilities
from utils.data_handler import load_dataset, create_data_loaders
from utils.visualization import plot_training_curves, plot_lambda_history, plot_sparsity
from utils.training_utils import evaluate_model, create_summary_table, save_model, load_model, calculate_overall_sparsity


def get_vgg_model(num_classes=14, pretrained=True):
    """Get a VGG11 model with a custom classifier."""
    model = models.vgg11(pretrained=pretrained)
    
    # Replace the classifier with our custom sequence
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
        nn.Softmax(dim=1)
    )
    
    return model


def train_base_model(model, train_loader, val_loader, num_epochs=10, device='cuda', 
                    output_dir='results/base_model', use_gpu_dataset=False):
    """Train a model without SABR regularization."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Move model to device
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Initialize tracking variables
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs = []
    lr_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_pbar:
            # Only transfer to device if not using GPU dataset
            if not use_gpu_dataset:
                inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with current batch statistics
            batch_loss = loss.item()
            batch_acc = predicted.eq(labels).sum().item() / labels.size(0)
            train_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}'
            })
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                # Only transfer to device if not using GPU dataset
                if not use_gpu_dataset:
                    inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar with current batch statistics
                batch_loss = loss.item()
                batch_acc = predicted.eq(labels).sum().item() / labels.size(0)
                val_pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, os.path.join(output_dir, 'best_model.pth'))
            print(f'✓ New best model saved with val_acc: {val_acc:.4f}')
        
        # Store metrics for plotting
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        lr_history.append(current_lr)
    
    print(f'\nBase model training completed. Best validation accuracy: {best_val_acc:.4f}')
    
    # Save training history
    history_df = pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': train_losses,
        'Val Loss': val_losses,
        'Train Acc': train_accs,
        'Val Acc': val_accs,
        'Learning Rate': lr_history
    })
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Plot and save training curves
    plot_training_curves(epochs, train_losses, val_losses, train_accs, val_accs, lr_history, output_dir)
    create_summary_table(history_df, output_dir)
    
    return model, best_val_acc


def train_with_sabr(model, train_loader, val_loader, num_epochs=40, device='cuda', 
                   output_dir='results/sabr_model', use_gpu_dataset=False,
                   teta1=5e-4, accuracy_threshold=0.007, growth_rate=1.15, decrease_rate=1.2, k_epoch=2):
    """Train a model with SABR regularization."""
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Move model to device
    model.to(device)
    
    # Set dropout to 0.0 (eliminate dropout)
    eliminate_dropout(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.51, patience=3, verbose=True, min_lr=0.00001)
    
    # Initialize the SABR regularizer
    regularizer = SABR(
        model,
        teta1=teta1,
        growth_rate=growth_rate,
        decrease_rate=decrease_rate,
        accuracy_threshold=accuracy_threshold,
        k_epoch=k_epoch
    )
    
    # Initialize tracking variables
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs = []
    lr_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Check if all parameters are frozen
        if regularizer.all_parameters_frozen():
            print("All parameters have been frozen. Stopping training.")
            break
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_pbar:
            # Only transfer to device if not using GPU dataset
            if not use_gpu_dataset:
                inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Add L1 penalty from the regularizer
            l1_penalty = regularizer.get_penalty(model)
            loss = loss + l1_penalty
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with current batch statistics
            batch_loss = loss.item()
            batch_acc = predicted.eq(labels).sum().item() / labels.size(0)
            train_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}'
            })
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                # Only transfer to device if not using GPU dataset
                if not use_gpu_dataset:
                    inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar with current batch statistics
                batch_loss = loss.item()
                batch_acc = predicted.eq(labels).sum().item() / labels.size(0)
                val_pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update lambda based on validation accuracy
        lambda_values = regularizer.update(val_acc)
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Learning Rate: {current_lr:.10f}')
        
        # Print lambda values for key layers (optional, for monitoring)
        print("Lambda values for sample layers:")
        for i, (layer_name, lambda_val) in enumerate(lambda_values.items()):
            if i < 3 or i > len(lambda_values) - 4:  # Show first 3 and last 3 layers
                frozen_status = " (FROZEN)" if layer_name in regularizer.get_frozen_parameters() else ""
                print(f'  {layer_name}: {lambda_val:.10f}{frozen_status}')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, os.path.join(output_dir, 'best_model.pth'))
            print(f'✓ New best model saved with val_acc: {val_acc:.4f}')
        
        # Store metrics for plotting
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        lr_history.append(current_lr)
        
        # Calculate and plot sparsity
        sparsity_dict = calculate_sparsity(model)
        plot_sparsity(sparsity_dict, epoch + 1, plots_dir)
        
        # Print overall sparsity
        overall_sparsity, total_params, zero_params = calculate_overall_sparsity(model)
        print(f"Overall model sparsity: {overall_sparsity:.2f}% ({zero_params}/{total_params} weights near zero)")
    
    print(f'\nSABR training completed. Best validation accuracy: {best_val_acc:.4f}')
    
    # Save final model if it's the last epoch
    save_model(model, os.path.join(output_dir, 'final_model.pth'))
    
    # Save training history
    history_df = pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': train_losses,
        'Val Loss': val_losses,
        'Train Acc': train_accs,
        'Val Acc': val_accs,
        'Learning Rate': lr_history
    })
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Plot and save training curves
    plot_training_curves(epochs, train_losses, val_losses, train_accs, val_accs, lr_history, output_dir)
    create_summary_table(history_df, output_dir)
    
    # Plot lambda histories
    plot_lambda_history(regularizer.get_lambda_history(), regularizer.get_frozen_parameters(), plots_dir)
    
    return model, best_val_acc


def apply_pruning(model, test_loader, output_dir='results/pruned_model', device='cuda',
                 use_gpu_dataset=False, use_std_based=True, teta1=0.2, gamma=0.1, epsilon=5e-4):
    """Apply post-training pruning to a model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Move model to device
    model.to(device)
    
    # Define loss function for evaluation
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate original model
    print("\nEvaluating original model...")
    original_loss, original_acc = evaluate_model(model, test_loader, criterion, device, use_gpu_dataset)
    print(f'Original Model - Test Loss: {original_loss:.4f}, Test Accuracy: {original_acc:.4f}')
    
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
    print(f"Overall model sparsity: {overall_sparsity:.2f}%")
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
        f.write(f"Overall model sparsity: {overall_sparsity:.2f}%\n")
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
    
    return pruned_model, pruned_acc






if __name__ == "__main__":
    """Run the complete SABR training pipeline."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Configuration
    data_dir = 'C:/Users/orani/bilel/a_miv/a_miv/m1s2/rnna/tp/project/deep_sp/dataset/HG14/HG14-Hand Gesture'  # Update this path
    output_dir = 'results'
    batch_size = 64
    image_size = 128
    use_gpu_dataset = True  # Set to True to load dataset into GPU memory
    
    # Create output directories
    base_dir = os.path.join(output_dir, 'base_model')
    sabr_dir = os.path.join(output_dir, 'sabr_model')
    pruned_dir = os.path.join(output_dir, 'pruned_model')
    
    # Load dataset
    print("\nLoading dataset...")
    image_paths, labels = load_dataset(data_dir, dataset_type='HG14')
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        image_paths, labels, batch_size=batch_size, image_size=image_size,
        device=device, use_gpu_dataset=use_gpu_dataset
    )
    
    # Phase 1: Base model training
    print("\n" + "="*50)
    print("PHASE 1: BASE MODEL TRAINING")
    print("="*50)
    
    # Initialize model
    model = get_vgg_model(num_classes=14, pretrained=True)
    
    # Train the base model
    print("Training base model...")
    base_model, base_val_acc = train_base_model(
        model, train_loader, val_loader, num_epochs=15, 
        device=device, output_dir=base_dir, use_gpu_dataset=use_gpu_dataset
    )
    
    # Phase 2: SABR regularization training
    print("\n" + "="*50)
    print("PHASE 2: SABR REGULARIZATION TRAINING")
    print("="*50)
    
    # Load the best model from Phase 1
    print("Loading best model from Phase 1...")
    load_model(model, os.path.join(base_dir, 'best_model.pth'), device)
    
    # Train with SABR regularization
    print("Training with SABR regularization...")
    sabr_model, sabr_val_acc = train_with_sabr(
        model, train_loader, val_loader, num_epochs=40, 
        device=device, output_dir=sabr_dir, use_gpu_dataset=use_gpu_dataset,
        teta1=5e-4, accuracy_threshold=0.007, growth_rate=1.15, decrease_rate=1.2, k_epoch=2
    )
    
    # Phase 3: Post-training pruning
    print("\n" + "="*50)
    print("PHASE 3: POST-TRAINING PRUNING")
    print("="*50)
    
    # Load the best model from Phase 2
    print("Loading best model from Phase 2...")
    load_model(model, os.path.join(sabr_dir, 'best_model.pth'), device)
    
    # Apply post-training pruning
    print("Applying post-training pruning...")
    pruned_model, pruned_acc = apply_pruning(
        model, test_loader, output_dir=pruned_dir,
        device=device, use_gpu_dataset=use_gpu_dataset,
        use_std_based=True, teta1=0.2, gamma=0.1
    )
    
    print("\nComplete training pipeline finished!")
    print(f"Base model (validation accuracy: {base_val_acc:.4f}) saved to: {base_dir}")
    print(f"SABR model (validation accuracy: {sabr_val_acc:.4f}) saved to: {sabr_dir}")
    print(f"Pruned model (test accuracy: {pruned_acc:.4f}) saved to: {pruned_dir}")
