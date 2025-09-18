"""
Inference example for ALGA-DMA cloud classification model

This script demonstrates how to use the trained model for inference on new images.

Author: Your Name
License: MIT
"""

import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import create_model
from data_utils import get_data_transforms, CLOUD_CATEGORIES


class CloudClassifier:
    """Cloud classification inference class"""
    
    def __init__(self, model_path, num_classes=7, device=None):
        """
        Initialize the classifier
        
        Args:
            model_path (str): Path to trained model weights
            num_classes (int): Number of cloud classes
            device: PyTorch device (auto-detected if None)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        # Load model
        self.model = create_model(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        transforms_dict = get_data_transforms()
        self.transform = transforms_dict['val']  # Use validation transforms for inference
        
        # Class names
        self.class_names = list(CLOUD_CATEGORIES.values())[:num_classes]
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {self.class_names}")
    
    def predict_single_image(self, image_path, return_probabilities=False):
        """
        Predict cloud type for a single image
        
        Args:
            image_path (str): Path to image file
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Predict cloud types for multiple images
        
        Args:
            image_paths (list): List of image file paths
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path, return_probabilities)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image and probabilities
        
        Args:
            image_path (str): Path to image file
            save_path (str): Path to save visualization (optional)
        """
        # Get prediction
        result = self.predict_single_image(image_path, return_probabilities=True)
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title(f"Predicted: {result['predicted_label']}\nConfidence: {result['confidence']:.3f}")
        ax1.axis('off')
        
        # Show probabilities
        classes = list(result['probabilities'].keys())
        probs = list(result['probabilities'].values())
        
        bars = ax2.barh(classes, probs)
        ax2.set_xlabel('Probability')
        ax2.set_title('Class Probabilities')
        ax2.set_xlim(0, 1)
        
        # Highlight predicted class
        predicted_idx = classes.index(result['predicted_label'])
        bars[predicted_idx].set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Example usage"""
    # Configuration
    model_path = "../models/best_model.pth"  # Update this path
    test_image_path = "../data/sample_image.jpg"  # Update this path
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train a model first or update the model path.")
        return
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please provide a test image or update the image path.")
        return
    
    # Initialize classifier
    classifier = CloudClassifier(model_path)
    
    # Single image prediction
    print("\n" + "="*50)
    print("Single Image Prediction")
    print("="*50)
    
    result = classifier.predict_single_image(test_image_path, return_probabilities=True)
    
    print(f"Image: {test_image_path}")
    print(f"Predicted Class: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nAll Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.3f}")
    
    # Visualize prediction
    print("\nGenerating visualization...")
    classifier.visualize_prediction(test_image_path, "prediction_result.png")
    
    # Batch prediction example
    print("\n" + "="*50)
    print("Batch Prediction Example")
    print("="*50)
    
    # Example with multiple images (update paths as needed)
    image_paths = [test_image_path]  # Add more paths here
    
    batch_results = classifier.predict_batch(image_paths, return_probabilities=True)
    
    for i, result in enumerate(batch_results):
        if 'error' in result:
            print(f"Image {i+1}: Error - {result['error']}")
        else:
            print(f"Image {i+1}: {result['predicted_label']} (confidence: {result['confidence']:.3f})")


if __name__ == "__main__":
    main()