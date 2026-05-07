"""
Testing Utilities for SABR Notebooks

This module contains functions for testing and evaluating models trained with SABR.
"""

import os
import torch
import random
from PIL import Image, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt

def load_test_samples(data_dir, num_samples_per_class=1, seed=42):
    """Load test samples from each class."""
    samples = []
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # For each gesture class
    for gesture_id in range(14):
        gesture_dir = os.path.join(data_dir, f"Gesture_{gesture_id}")
        
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(gesture_dir) if f.endswith('.jpg')]
        
        # Randomly select samples
        selected_files = random.sample(image_files, min(num_samples_per_class, len(image_files)))
        
        # Add selected samples to the list
        for file_name in selected_files:
            samples.append({
                'path': os.path.join(gesture_dir, file_name),
                'class_id': gesture_id,
                'class_name': f"Gesture {gesture_id}"
            })
    
    return samples

def preprocess_image(image_path, image_size=128):
    """Preprocess an image for model input."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = ImageOps.exif_transpose(image)
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor, image

def predict_image(model, image_tensor, device='cuda'):
    """Run prediction on an image tensor."""
    model.eval()
    with torch.no_grad():
        # Move tensor to device
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item()
    
    return prediction, confidence, probabilities.cpu().numpy()

def plot_prediction(image, true_class, prediction, confidence, probabilities, save_path=None):
    """Plot the image with prediction results."""
    plt.figure(figsize=(12, 6))
    
    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"True Class: {true_class}")
    plt.axis('off')
    
    # Create color for the prediction (green if correct, red if wrong)
    correct = true_class.split()[-1] == str(prediction)
    color = 'green' if correct else 'red'
    
    # Add prediction text
    plt.figtext(0.5, 0.01, f"Prediction: Gesture {prediction} (Confidence: {confidence:.2f})", 
                ha='center', fontsize=12, bbox={'facecolor': color, 'alpha': 0.2, 'pad': 5})
    
    # Plot probability distribution
    plt.subplot(1, 2, 2)
    plt.bar(range(len(probabilities)), probabilities)
    plt.title('Class Probabilities')
    plt.xlabel('Gesture Class')
    plt.ylabel('Probability')
    plt.xticks(range(len(probabilities)))
    plt.ylim(0, 1)
    
    # Highlight the true class and predicted class
    true_class_id = int(true_class.split()[-1])
    plt.bar(true_class_id, probabilities[true_class_id], color='blue', label='True Class')
    if prediction != true_class_id:
        plt.bar(prediction, probabilities[prediction], color='red', label='Predicted Class')
    plt.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()

def test_model(model, data_dir, test_dir, device='cuda', image_size=128, num_samples_per_class=2):
    """Test a model on sample images from the dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)
    
    # Load test samples
    test_samples = load_test_samples(data_dir, num_samples_per_class)
    
    # Make sure model is in evaluation mode
    model.eval()
    
    # Test results
    results = []
    
    # Test each sample
    for i, sample in enumerate(test_samples):
        # Preprocess image
        image_tensor, original_image = preprocess_image(sample['path'], image_size)
        
        # Run prediction
        prediction, confidence, probabilities = predict_image(model, image_tensor, device)
        
        # Plot prediction
        save_path = os.path.join(test_dir, f"test_sample_{i+1}_class_{sample['class_id']}.png")
        plot_prediction(original_image, sample['class_name'], prediction, confidence, probabilities, save_path)
        
        # Store result
        correct = prediction == sample['class_id']
        results.append({
            'sample_id': i+1,
            'true_class': sample['class_name'],
            'predicted_class': f"Gesture {prediction}",
            'confidence': confidence,
            'correct': correct
        })
        
        # Print result
        result_str = "✓ Correct" if correct else "✗ Incorrect"
        print(f"Sample {i+1}: {result_str} | True: {sample['class_name']} | Predicted: Gesture {prediction} | Confidence: {confidence:.2f}")
    
    # Calculate accuracy
    accuracy = sum(r['correct'] for r in results) / len(results)
    print(f"\nTest accuracy on {len(results)} samples: {accuracy:.2f}")
    
    return results, accuracy
