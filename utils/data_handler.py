"""
Data Handling Utilities for SABR

This module contains functions and classes for handling datasets,
particularly for the HG14 hand gesture dataset.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class HG14Dataset(Dataset):
    """Dataset class for HG14 Hand Gesture dataset."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # Apply EXIF orientation using ImageOps
        image = ImageOps.exif_transpose(image)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class DatasetGPU(Dataset):
    """
    Memory-optimized dataset class that loads all images into GPU memory.
    """
    def __init__(self, image_paths, labels, transform=None, device='cuda'):
        self.device = device
        
        # Process all images and load them into GPU memory at once
        print("Loading all images into GPU memory...")
        images = []
        for img_path in tqdm(image_paths, desc="Processing images"):
            image = Image.open(img_path).convert('RGB')
            image = ImageOps.exif_transpose(image)
            
            if transform:
                image = transform(image)
            
            images.append(image)
        
        # Convert to tensor and move to GPU
        self.images = torch.stack(images).to(device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)
        
        # Report memory usage
        memory_used = self.images.element_size() * self.images.nelement() / 1024 / 1024  # in MB
        print(f"Dataset loaded to GPU. Images shape: {self.images.shape}, Memory used: {memory_used:.2f} MB")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # No need to transfer to device since data is already on GPU
        return self.images[idx], self.labels[idx]


def get_transforms(image_size=224):
    """
    Get data transforms for training and validation/testing.
    
    Args:
        image_size: Size to resize the images to
        
    Returns:
        Dictionary containing 'train' and 'val_test' transforms
    """
    # Define transforms with enhanced augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.8, 1.0),  # Random scale between 80-100% of original size
            ratio=(0.9, 1.1)   # Slight aspect ratio changes
        ),
        transforms.ColorJitter(
            brightness=0.2,    # Random brightness adjustment
            contrast=0.2,      # Random contrast adjustment
            saturation=0.1,    # Slight saturation changes
            hue=0.05           # Very slight hue shifts (hand color should remain natural)
        ),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
    ])
    
    # Simpler transform for validation and testing
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val_test': val_test_transform
    }


def load_dataset(data_dir, dataset_type='HG14'):
    """
    Load image paths and labels from a dataset directory.
    
    Args:
        data_dir: Path to the dataset directory
        dataset_type: Type of dataset (currently only 'HG14' is supported)
        
    Returns:
        image_paths: List of image file paths
        labels: List of labels corresponding to the images
    """
    image_paths = []
    labels = []
    
    if dataset_type == 'HG14':
        # Show progress when loading dataset
        print("Loading HG14 dataset paths...")
        for gesture_id in tqdm(range(14), desc="Loading gesture classes"):
            gesture_dir = os.path.join(data_dir, f"Gesture_{gesture_id}")
            files = [f for f in os.listdir(gesture_dir) if f.endswith('.jpg')]
            for img_name in files:
                img_path = os.path.join(gesture_dir, img_name)
                image_paths.append(img_path)
                labels.append(gesture_id)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    print(f"Total images found: {len(image_paths)}")
    return image_paths, labels


def create_data_loaders(image_paths, labels, batch_size=32, image_size=224, 
                        device='cuda', use_gpu_dataset=False):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        image_paths: List of image file paths
        labels: List of labels corresponding to the images
        batch_size: Batch size for the data loaders
        image_size: Size to resize the images to
        device: Device to use ('cuda' or 'cpu')
        use_gpu_dataset: Whether to use the GPU-optimized dataset
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
    """
    # Get transforms
    transforms_dict = get_transforms(image_size)
    train_transform = transforms_dict['train']
    val_test_transform = transforms_dict['val_test']
    
    # First, split out 10% for testing
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=0.10, random_state=42, stratify=labels
    )
    
    # Then split the remaining 90% into training (80% of remaining) and validation (20% of remaining)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val
    )
    
    print(f"Split sizes - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets - choose between GPU or regular dataset based on parameter
    if use_gpu_dataset:
        print("Creating GPU-loaded datasets...")
        train_dataset = DatasetGPU(X_train, y_train, transform=train_transform, device=device)
        val_dataset = DatasetGPU(X_val, y_val, transform=val_test_transform, device=device)
        test_dataset = DatasetGPU(X_test, y_test, transform=val_test_transform, device=device)
        
        # For GPU datasets, use num_workers=0 and pin_memory=False
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    else:
        print("Creating regular CPU datasets...")
        train_dataset = HG14Dataset(X_train, y_train, transform=train_transform)
        val_dataset = HG14Dataset(X_val, y_val, transform=val_test_transform)
        test_dataset = HG14Dataset(X_test, y_test, transform=val_test_transform)
        
        # For CPU datasets, use standard settings
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader
