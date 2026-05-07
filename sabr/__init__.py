"""
SABR (Sparsity-Accuracy Balanced Regularization) Package

Core algorithm for balancing model sparsity and accuracy through adaptive regularization.
"""

from .sabr import SABR
from .lambda_calculator import calculate_teta1_std_values
from .pruning import prune_model, calculate_sparsity, eliminate_dropout

__all__ = ['SABR', 'calculate_teta1_std_values', 'prune_model', 'calculate_sparsity', 'eliminate_dropout']
