"""
SABR (Sparsity-Accuracy Balanced Regularization) Algorithm

Core implementation of the SABR algorithm for neural network pruning.
"""

import torch
from .lambda_calculator import calculate_teta1_std_values


class SABR:
    """
    SABR (Sparsity-Accuracy Balanced Regularization) algorithm that 
    automatically adjusts lambda parameters based on validation performance
    to maintain model performance while increasing regularization.
    
    This class uses layer-specific lambda values initialized based on teta1 * std(w)
    for each layer for adaptive regularization control.
    """
    
    def __init__(self, model, teta1=5e-4, growth_rate=1.1, decrease_rate=1.2, 
                 accuracy_threshold=0.005, k_epoch=3):
        """
        Initialize the SABR algorithm with layer-specific lambda values.
        
        Args:
            model: PyTorch model to regularize
            teta1: Parameter to multiply with std deviation of each layer
            growth_rate: Factor by which to increase lambda (e.g., 1.1 means 10% increase)
            decrease_rate: Factor by which to decrease lambda (e.g., 1.2 means divide by 1.2)
            accuracy_threshold: Maximum acceptable drop in validation accuracy from best
            k_epoch: Number of epochs without lambda growth before freezing the most sparse parameter
        """
        # Calculate initial lambda values for each layer using teta1 * std
        self.lambda_values = calculate_teta1_std_values(model, teta1)
        self.initial_lambda_values = self.lambda_values.copy()
        
        self.growth_rate = growth_rate
        self.decrease_rate = decrease_rate
        self.accuracy_threshold = accuracy_threshold
        self.max_val_acc = None
        self.k_epoch = k_epoch
        
        # Track epochs without lambda growth for each parameter
        self.no_growth_epochs = {layer_name: 0 for layer_name in self.lambda_values.keys()}
        
        # Set to keep track of parameters with frozen lambdas
        self.frozen_lambdas = set()
        
        # Initialize lambda history for each layer
        self.lambda_history = {layer_name: [lambda_val] for layer_name, lambda_val in self.lambda_values.items()}
        
        # Store model reference for sparsity calculations
        self.model = model
        
        # Print initial lambda values
        print(f"Initialized SABR with teta1={teta1}")
        for layer_name, lambda_val in self.lambda_values.items():
            print(f"  {layer_name}: initial lambda = {lambda_val:.8f}")
    
    def get_penalty(self, model):
        """
        Calculate the L1 penalty for the model's parameters.
        Each layer uses its own specific lambda value.
        
        Args:
            model: PyTorch model
            
        Returns:
            L1 penalty tensor that can be added to the loss
        """
        l1_penalty = 0
        for name, param in model.named_parameters():
            if name in self.lambda_values:
                l1_penalty += self.lambda_values[name] * torch.sum(torch.abs(param))
        
        return l1_penalty
    
    def find_parameter_to_freeze(self):
        """
        Identifies which parameter should have its lambda frozen based on
        the number of non-zero elements. Parameters with fewest non-zero
        elements (most sparse) will be selected first.
        
        Returns:
            String: name of the parameter to freeze
        """
        # Dictionary to store parameter name and count of non-zero elements
        non_zero_counts = {}
        
        # Calculate non-zero elements for each parameter
        for name, param in self.model.named_parameters():
            if name in self.lambda_values and name not in self.frozen_lambdas:
                # Get the weights as a numpy array
                weights = param.data.cpu().numpy()
                # Count non-zero elements
                non_zero_count = weights.size - (abs(weights) < 1e-5).sum()
                non_zero_counts[name] = non_zero_count
        
        # If all lambdas are already frozen or no weights to check, return None
        if not non_zero_counts:
            return None
        
        # Find the parameter with the minimum number of non-zero elements
        min_param = min(non_zero_counts.items(), key=lambda x: x[1])
        param_name = min_param[0]
        
        # Set requires_grad to False for this parameter
        for name, param in self.model.named_parameters():
            if name == param_name:
                param.requires_grad = False
                print(f"Setting {param_name} to non-trainable (requires_grad=False)")
                break
                
        return param_name
    
    def update(self, current_val_acc):
        """
        Update all lambda values based on validation accuracy change compared to max.
        This should be called once per epoch after validation.
        
        Args:
            current_val_acc: Current epoch's validation accuracy
            
        Returns:
            lambda_values: Dictionary of updated lambda values
        """
        # Initialize max_val_acc if first epoch
        if self.max_val_acc is None:
            self.max_val_acc = current_val_acc
        
        # Calculate accuracy drop from best historical performance
        accuracy_drop = self.max_val_acc - current_val_acc
        
        # Flag to track if we increased lambdas in this update
        increased_lambdas = False
        
        # Get count of unfrozen parameters
        unfrozen_params = [name for name in self.lambda_values if name not in self.frozen_lambdas]
        unfrozen_count = len(unfrozen_params)
        
        # Update lambda based on accuracy drop
        if accuracy_drop > self.accuracy_threshold:
            # Reduce lambda if accuracy dropped too much from best
            for layer_name in unfrozen_params:
                self.lambda_values[layer_name] /= self.decrease_rate
                # DO NOT reset no-growth counter here - we're not growing
            
            print(f"Accuracy dropped by {accuracy_drop:.4f} > threshold {self.accuracy_threshold} from max {self.max_val_acc:.4f}. "
                  f"Reducing all lambda values by factor of {1/self.decrease_rate:.3f}")
        else:
            # Increase lambda if accuracy is stable compared to best
            if unfrozen_count > 0:
                for layer_name in unfrozen_params:
                    self.lambda_values[layer_name] *= self.growth_rate
                    self.no_growth_epochs[layer_name] = 0  # Reset counter as we're growing
                increased_lambdas = True
                
                print(f"Accuracy stable (drop: {accuracy_drop:.4f} <= threshold {self.accuracy_threshold} from max {self.max_val_acc:.4f}). "
                      f"Increasing all lambda values by factor of {self.growth_rate:.3f}")
            else:
                print(f"Accuracy stable but all parameters are frozen. No lambda updates applied.")
        
        # If we didn't increase lambdas and have unfrozen parameters, increment the no-growth counters
        if not increased_lambdas and unfrozen_count > 0:
            # Increment no-growth counter for each unfrozen parameter
            for layer_name in unfrozen_params:
                self.no_growth_epochs[layer_name] += 1
            
            # Check if any parameter has reached k_epoch without growth
            parameters_at_limit = [name for name in unfrozen_params 
                                  if self.no_growth_epochs[name] >= self.k_epoch]
            
            if parameters_at_limit:
                # Find parameter to freeze based on non-zero element count
                param_to_freeze = self.find_parameter_to_freeze()
                
                if param_to_freeze:
                    self.frozen_lambdas.add(param_to_freeze)
                    print(f"Freezing lambda for {param_to_freeze} after {self.k_epoch} epochs without growth. "
                          f"Final lambda value: {self.lambda_values[param_to_freeze]:.8f}")
        
        # Update maximum validation accuracy if current is better
        if current_val_acc > self.max_val_acc:
            self.max_val_acc = current_val_acc
            print(f"New maximum validation accuracy: {self.max_val_acc:.4f}")
        
        # Update lambda history for each layer
        for layer_name, lambda_val in self.lambda_values.items():
            self.lambda_history[layer_name].append(lambda_val)
        
        # Print frozen parameters for tracking
        if self.frozen_lambdas:
            frozen_count = len(self.frozen_lambdas)
            total_count = len(self.lambda_values)
            print(f"Currently {frozen_count}/{total_count} parameters have frozen lambda values.")
        
        return self.lambda_values
    
    def all_parameters_frozen(self):
        """
        Check if all parameters have been frozen.
        
        Returns:
            bool: True if all parameters are frozen, False otherwise
        """
        return len(self.frozen_lambdas) == len(self.lambda_values)
    
    def get_current_lambda_values(self):
        """Get current lambda values for all layers."""
        return self.lambda_values
    
    def get_lambda_history(self):
        """Get history of lambda values for all layers."""
        return self.lambda_history
        
    def get_frozen_parameters(self):
        """Get the set of parameters that have been frozen."""
        return self.frozen_lambdas
