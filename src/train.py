"""
Training script for ALGA-DMA cloud classification model

This script handles the complete training pipeline including data loading,
model training, validation, and evaluation.

Author: Your Name
License: MIT
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
import logging
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from model import create_model
from data_utils import create_data_loaders, print_dataset_info


def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (data, targets) in enumerate(progress_bar):
        data, targets = data.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct_predictions / total_samples
    
    logger.info(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, logger):
    """Validate model for one epoch"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for data, targets in progress_bar:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct_predictions / total_samples
    
    logger.info(f"Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, device, num_epochs, logger, save_dir="models"):
    """Complete training loop"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 50)
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        # Validation phase
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, logger
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, checkpoint_path)
    
    logger.info(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    return history, best_model_path


def evaluate_model(model, test_loader, device, logger, num_classes=7):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    logger.info(f"\nTest Results:")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    
    # Detailed classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_targets, all_predictions))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    }


def plot_training_history(history, save_path="training_history.png"):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train ALGA-DMA Cloud Classification Model')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=7,
                       help='Number of cloud classes')
    parser.add_argument('--augment_data', action='store_true',
                       help='Generate augmented data')
    parser.add_argument('--augmented_path', type=str, default='augmented_data',
                       help='Path for augmented dataset')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for log files')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        augment_data=args.augment_data,
        augmented_path=args.augmented_path
    )
    
    # Print dataset information
    print_dataset_info(train_loader.dataset, "Training Set")
    print_dataset_info(val_loader.dataset, "Validation Set")
    print_dataset_info(test_loader.dataset, "Test Set")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(num_classes=args.num_classes)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)
    
    # Train model
    history, best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        logger=logger,
        save_dir=args.save_dir
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(args.save_dir, 'training_history.png'))
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_loader, device, logger, args.num_classes)
    
    # Save results
    results_path = os.path.join(args.save_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()