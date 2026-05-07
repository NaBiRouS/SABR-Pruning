"""
Real-time Hand Gesture Recognition using SABR-pruned Models

This script demonstrates real-time inference with a pruned model trained using SABR.
It captures webcam input and performs gesture recognition in real-time.
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image, ImageOps
import time
import argparse
import random


# Class names for the 14 gestures
GESTURE_CLASSES = [f"Gesture_{i}" for i in range(14)]


def get_vgg_model(num_classes=14, pretrained=False):
    """Get a VGG11 model with a custom classifier."""
    model = models.vgg11(pretrained=pretrained)
    
    # Replace the classifier with our custom sequence
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.0),  # No dropout for inference
        nn.Linear(512, num_classes),
        nn.Softmax(dim=1)
    )
    
    return model


def load_sample_images(dataset_dir, size=150):
    """
    Load sample images for each gesture class from the dataset.
    
    Args:
        dataset_dir: Path to the dataset directory containing gesture folders
        size: Size to resize the sample images to
    
    Returns:
        Dictionary mapping gesture IDs to sample images
    """
    sample_images = {}
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset directory {dataset_dir} not found. Sample images will not be displayed.")
        return sample_images
    
    # For each gesture class
    for gesture_id in range(14):
        gesture_dir = os.path.join(dataset_dir, f"Gesture_{gesture_id}")
        
        # Check if gesture directory exists
        if not os.path.exists(gesture_dir):
            print(f"Warning: Directory for Gesture_{gesture_id} not found.")
            continue
        
        # Get list of image files in the gesture directory
        image_files = [f for f in os.listdir(gesture_dir) if f.endswith('.jpg')]
        
        if not image_files:
            print(f"Warning: No image files found for Gesture_{gesture_id}.")
            continue
        
        # Randomly select one image from the class
        sample_file = random.choice(image_files)
        sample_path = os.path.join(gesture_dir, sample_file)
        
        # Load and resize the image
        try:
            # Use PIL for EXIF orientation handling
            pil_img = Image.open(sample_path).convert('RGB')
            pil_img = ImageOps.exif_transpose(pil_img)
            
            # Convert to OpenCV format for visualization
            sample_img = np.array(pil_img)
            sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR)
            
            # Resize the image
            sample_img = cv2.resize(sample_img, (size, size))
            
            # Add a border and gesture label
            cv2.rectangle(sample_img, (0, 0), (size-1, size-1), (0, 255, 0), 2)
            cv2.putText(sample_img, f"Gesture {gesture_id}", (5, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(sample_img, f"Gesture {gesture_id}", (5, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            sample_images[gesture_id] = sample_img
            print(f"Loaded sample for Gesture_{gesture_id}")
        except Exception as e:
            print(f"Error loading sample for Gesture_{gesture_id}: {str(e)}")
    
    return sample_images


def preprocess_frame(frame, image_size=224):
    """
    Preprocess a frame for input to the model.
    
    Args:
        frame: OpenCV frame (BGR)
        image_size: Size to resize the image to
        
    Returns:
        Tensor for model input
    """
    # Convert BGR (OpenCV format) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    # Apply EXIF orientation if needed (not usually needed for webcam frames, but included for consistency)
    pil_image = ImageOps.exif_transpose(pil_image)
    
    # Create the transform pipeline - must match validation transform from training
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform
    tensor = transform(pil_image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def draw_text(img, text, position, font_scale=0.7, thickness=2, 
              text_color=(255, 255, 255), bg_color=(0, 0, 0), padding=5):
    """
    Draw text with background on the image.
    
    Args:
        img: OpenCV image
        text: Text to draw
        position: (x, y) position to draw text
        font_scale: Font scale
        thickness: Text thickness
        text_color: Text color
        bg_color: Background color
        padding: Padding around text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate text position and background rectangle
    x, y = position
    bg_rect = (x, y - text_height - padding, 
               x + text_width + padding * 2, y + padding)
    
    # Draw background rectangle
    cv2.rectangle(img, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x + padding, y - padding), font, font_scale, text_color, thickness)
    
    return img


def main(image_size=128, model_path=None, camera_id=0, dataset_dir=None):
    """
    Main function for real-time gesture recognition.
    
    Args:
        image_size: Size to resize images to for the model
        model_path: Path to the trained model file
        camera_id: Camera device index
        dataset_dir: Path to the dataset directory for sample images
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # If no model path specified, use a default path
    if model_path is None:
        model_path = "results/pruned_model/pruned_model_std_based_teta1_0.2_gamma_0.1.pth"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    # Initialize model
    model = get_vgg_model(num_classes=14)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Load sample images from dataset
    sample_images = {}
    if dataset_dir:
        print(f"Loading sample images from {dataset_dir}...")
        sample_images = load_sample_images(dataset_dir, size=150)
        print(f"Loaded {len(sample_images)} sample images.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)  # Use specified camera_id
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("\nInstructions:")
    print("- Press 'q' or 'ESC' to quit")
    print("- Press 's' to save the current frame")
    print("- Press 'r' to toggle ROI mode (default: off)")
    
    # Initialize variables
    roi_mode = False
    roi_rect = None
    roi_dragging = False
    roi_point1 = None
    roi_point2 = None
    
    # For FPS calculation
    prev_time = time.time()
    fps_counter = 0
    fps = 0
    
    current_prediction = 0  # Default prediction
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_rect, roi_dragging, roi_point1, roi_point2
        
        if not roi_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_dragging = True
            roi_point1 = (x, y)
            roi_point2 = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if roi_dragging:
                roi_point2 = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            roi_dragging = False
            roi_point2 = (x, y)
            # Ensure points are in correct order (top-left, bottom-right)
            x1, y1 = min(roi_point1[0], roi_point2[0]), min(roi_point1[1], roi_point2[1])
            x2, y2 = max(roi_point1[0], roi_point2[0]), max(roi_point1[1], roi_point2[1])
            roi_rect = (x1, y1, x2, y2)
    
    # Set up mouse callback
    cv2.namedWindow('Hand Gesture Recognition')
    cv2.setMouseCallback('Hand Gesture Recognition', mouse_callback)
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
            
            # Mirror the frame horizontally for a more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                prev_time = current_time
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Process the ROI if in ROI mode
            if roi_mode and roi_rect:
                x1, y1, x2, y2 = roi_rect
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                roi_frame = frame[y1:y2, x1:x2]
                if roi_frame.size > 0:  # Check if ROI is valid
                    # Preprocess ROI for model input
                    input_tensor = preprocess_frame(roi_frame, image_size=image_size)
                    input_tensor = input_tensor.to(device)
                    
                    # Run inference
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                        prediction = torch.argmax(probabilities).item()
                        confidence = probabilities[prediction].item()
                        current_prediction = prediction  # Update current prediction
                    
                    # Display prediction
                    prediction_text = f"{GESTURE_CLASSES[prediction]} ({confidence:.2f})"
                    draw_text(display_frame, prediction_text, (10, 30))
                    
                    # Show the resized ROI
                    roi_resized = cv2.resize(roi_frame, (image_size, image_size))
                    roi_window_size = 150
                    roi_display = cv2.resize(roi_resized, (roi_window_size, roi_window_size))
                    display_frame[10:10+roi_window_size, display_frame.shape[1]-10-roi_window_size:display_frame.shape[1]-10] = roi_display
            
            # Process the whole frame if not in ROI mode
            else:
                # Preprocess frame for model input
                input_tensor = preprocess_frame(frame, image_size=image_size)
                input_tensor = input_tensor.to(device)
                
                # Run inference
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    prediction = torch.argmax(probabilities).item()
                    confidence = probabilities[prediction].item()
                    current_prediction = prediction  # Update current prediction
                
                # Display prediction
                prediction_text = f"{GESTURE_CLASSES[prediction]} ({confidence:.2f})"
                draw_text(display_frame, prediction_text, (10, 30))
            
            # Draw ROI instructions and mode indicator
            if roi_mode:
                mode_text = "ROI Mode: ON (Drag to select region)"
                if roi_dragging and roi_point1 and roi_point2:
                    cv2.rectangle(display_frame, roi_point1, roi_point2, (255, 0, 0), 2)
            else:
                mode_text = "ROI Mode: OFF (Press 'r' to enable)"
            
            # Display the sample image in the bottom-right corner
            if sample_images and current_prediction in sample_images:
                sample = sample_images[current_prediction]
                sample_size = sample.shape[0]
                sample_pos_y = display_frame.shape[0] - sample_size - 10  # 10px padding from bottom
                sample_pos_x = display_frame.shape[1] - sample_size - 10  # 10px padding from right
                
                # Create a region for the sample in the main display frame
                display_frame[sample_pos_y:sample_pos_y+sample_size, 
                             sample_pos_x:sample_pos_x+sample_size] = sample
            
            # Add labels and status information
            draw_text(display_frame, mode_text, (10, display_frame.shape[0] - 10))
            draw_text(display_frame, f"FPS: {fps}", (display_frame.shape[1] - 120, 30))
            
            # Show frame
            cv2.imshow('Hand Gesture Recognition', display_frame)
            
            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break
            elif key == ord('s'):  # 's' to save frame
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"gesture_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('r'):  # 'r' to toggle ROI mode
                roi_mode = not roi_mode
                if not roi_mode:
                    roi_rect = None
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Hand Gesture Recognition')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size for the model input (default: 128)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the trained model file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to the dataset directory containing gesture folders')
    
    args = parser.parse_args()
    
    # Run the application
    main(image_size=args.image_size, model_path=args.model, 
         camera_id=args.camera, dataset_dir=args.dataset)
