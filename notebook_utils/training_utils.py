"""
Training Utilities for SABR Notebooks

This module contains training functions for SABR notebooks.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np

# Import SABR components
from sabr.sabr import SABR
from sabr.pruning import prune_model, eliminate_dropout, calculate_sparsity

# Import utilities
from utils.training_utils import evaluate_model, save_model, load_model, calculate_overall_sparsity

# Import visualization utilities
from notebook_utils.visualization_utils import (
    plot_training_curves, plot_lambda_history, plot_sparsity, 
    plot_weight_distributions, plot_sparsity_evolution,
    create_summary_table, plot_computational_savings,
    create_final_summary_table
)
