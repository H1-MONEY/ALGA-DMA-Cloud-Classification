"""
Data utilities for cloud classification

This module contains functions for data loading, preprocessing, and augmentation.

Author: Your Name
License: MIT
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import cv2
import shutil
from collections import Counter


# Cloud category mapping
CLOUD_CATEGORIES = {
    '1_cumulus': 'Cumulus',
    '2_altocumulus': 'Altocumulus', 
    '3_cirrus': 'Cirrus',
    '4_clearsky': 'Clear Sky',
    '5_stratocumulus': 'Stratocumulus',
    '6_cumulonimbus': 'Cumulonimbus',
    '7_mixed': 'Mixed Cloud'
}


def get_data_transforms(image_size=256):
    """
    Get data transformation pipelines for training and validation
    
    Args:
        image_size (int): Target image size
        
    Returns:
        dict: Dictionary containing train and validation transforms
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms


def apply_color_jitter_cv2(img, brightness=0, contrast=0, saturation=0, hue=0):
    """
    Apply color jittering using OpenCV
    
    Args:
        img: Input image (PIL or numpy array)
        brightness: Brightness adjustment factor
        contrast: Contrast adjustment factor  
        saturation: Saturation adjustment factor
        hue: Hue adjustment factor
        
    Returns:
        numpy.ndarray: Processed image
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convert to HSV for hue and saturation adjustments
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Adjust hue
    if hue != 0:
        hsv[:, :, 0] = (hsv[:, :, 0] + hue * 180) % 180
    
    # Adjust saturation
    if saturation != 0:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation), 0, 255)
    
    # Convert back to RGB
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Adjust brightness and contrast
    if brightness != 0 or contrast != 0:
        img = img.astype(np.float32)
        img = img * (1 + contrast) + brightness * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img


def generate_augmented_images(source_path, target_path, num_variants=4):
    """
    Generate augmented images with different intensity levels
    
    Args:
        source_path (str): Path to source dataset
        target_path (str): Path to save augmented dataset
        num_variants (int): Number of augmentation variants per image
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        
    for class_name in os.listdir(source_path):
        if class_name.startswith('.'):
            continue
            
        class_dir = os.path.join(source_path, class_name)
        target_class_dir = os.path.join(target_path, class_name)
        
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)
            
        for image_filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_filename)
            
            if os.path.isfile(image_path) and image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    image = Image.open(image_path).convert('RGB')
                    
                    # Define different strength levels
                    strength_levels = [0.1, 0.2, 0.3, 0.4]
                    
                    for strength_idx, strength in enumerate(strength_levels[:num_variants]):
                        # Create transform with current strength
                        transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ColorJitter(
                                brightness=strength, 
                                contrast=strength, 
                                saturation=strength, 
                                hue=strength
                            ),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        
                        # Apply transformation
                        transformed_tensor = transform(image)
                        
                        # Convert back to PIL for saving
                        # Denormalize first
                        denorm_transform = transforms.Compose([
                            transforms.Normalize(
                                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                std=[1/0.229, 1/0.224, 1/0.225]
                            ),
                            transforms.ToPILImage()
                        ])
                        
                        transformed_image = denorm_transform(transformed_tensor)
                        
                        # Save augmented image
                        base_name = os.path.splitext(image_filename)[0]
                        save_path = os.path.join(
                            target_class_dir, 
                            f"{base_name}_aug_{strength_idx+1}.jpg"
                        )
                        transformed_image.save(save_path)
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")


def stratified_split(dataset, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42):
    """
    Perform stratified split of dataset
    
    Args:
        dataset: PyTorch dataset
        train_size (float): Proportion for training
        val_size (float): Proportion for validation
        test_size (float): Proportion for testing
        random_state (int): Random seed
        
    Returns:
        tuple: Train, validation, and test indices
    """
    # Extract labels from dataset
    if hasattr(dataset, 'samples'):
        labels = [s[1] for s in dataset.samples]
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        raise ValueError("Dataset must have 'samples' or 'targets' attribute")
    
    # First split: separate train from (val + test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=val_size + test_size, 
        random_state=random_state
    )
    train_idx, temp_idx = next(sss1.split(range(len(labels)), labels))
    
    # Second split: separate val from test
    temp_labels = [labels[i] for i in temp_idx]
    val_ratio = val_size / (val_size + test_size)
    
    sss2 = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=1-val_ratio, 
        random_state=random_state
    )
    val_idx_temp, test_idx_temp = next(sss2.split(range(len(temp_idx)), temp_labels))
    
    # Map back to original indices
    val_idx = temp_idx[val_idx_temp]
    test_idx = temp_idx[test_idx_temp]
    
    return train_idx, val_idx, test_idx


def create_data_loaders(dataset_path, batch_size=32, num_workers=4, 
                       train_size=0.7, val_size=0.2, test_size=0.1,
                       augment_data=True, augmented_path=None):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        dataset_path (str): Path to dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes
        train_size (float): Training set proportion
        val_size (float): Validation set proportion  
        test_size (float): Test set proportion
        augment_data (bool): Whether to generate augmented data
        augmented_path (str): Path to save/load augmented data
        
    Returns:
        tuple: Train, validation, and test data loaders
    """
    # Generate augmented data if requested
    if augment_data and augmented_path:
        if not os.path.exists(augmented_path):
            print("Generating augmented dataset...")
            generate_augmented_images(dataset_path, augmented_path)
        dataset_path = augmented_path
    
    # Get transforms
    transforms_dict = get_data_transforms()
    
    # Load dataset
    full_dataset = datasets.ImageFolder(dataset_path, transform=transforms_dict['train'])
    
    # Perform stratified split
    train_idx, val_idx, test_idx = stratified_split(
        full_dataset, train_size, val_size, test_size
    )
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    # Update validation dataset transform
    val_dataset.dataset.transform = transforms_dict['val']
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def print_dataset_info(dataset, name="Dataset"):
    """
    Print information about dataset distribution
    
    Args:
        dataset: PyTorch dataset
        name (str): Name for display
    """
    if hasattr(dataset, 'dataset'):
        # Handle Subset datasets
        indices = dataset.indices
        if hasattr(dataset.dataset, 'samples'):
            labels = [dataset.dataset.samples[i][1] for i in indices]
        else:
            labels = [dataset.dataset.targets[i] for i in indices]
    else:
        # Handle regular datasets
        if hasattr(dataset, 'samples'):
            labels = [s[1] for s in dataset.samples]
        else:
            labels = dataset.targets
    
    label_counts = Counter(labels)
    print(f"\n{name} distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} samples")
    print(f"  Total: {len(labels)} samples")


if __name__ == "__main__":
    # Test data loading
    print("Testing data utilities...")
    
    # Example usage
    dataset_path = "path/to/your/dataset"
    if os.path.exists(dataset_path):
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_path, 
            batch_size=16,
            augment_data=False
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
    else:
        print("Dataset path not found. Please update the path.")