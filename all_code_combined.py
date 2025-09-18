# ============================================================================
# ALGA Cloud Classification - All Code Combined
# ============================================================================
# This file contains all the code from the ALGA Cloud Classification project
# combined into a single file for easy reference and deployment.
#
# Project: ALGA (Adaptive Local-Global Attention) for Cloud Classification
# Description: A deep learning model combining DenseNet, Vision Transformer, 
#              and Dynamic Multi-Head Attention for cloud image classification
# ============================================================================

# ALGA Cloud Classification Model
# A deep learning model for cloud classification using ALGA (Adaptive Local-Global Attention) mechanism
# Combined with DenseNet, Vision Transformer, and Dynamic Multi-Head Attention

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.models import densenet121, vit_b_16, ViT_B_16_Weights
import matplotlib.pyplot as plt
from PIL import Image
import math
import sys
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import shutil
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

logging.basicConfig(filename='training_log.log', level=logging.INFO)

# Set GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DATA PROCESSING AND AUGMENTATION
# ============================================================================

# Data augmentation function
def generate_augmented_images(source_path, target_path, num_variants=5, color_jitter_strength=None):
    """Generate augmented images with different color jitter strengths"""
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for class_name in os.listdir(source_path):
        if class_name.startswith('.'):
            continue  # Skip hidden folders
        class_dir = os.path.join(source_path, class_name)
        target_class_dir = os.path.join(target_path, class_name)
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)
        for image_filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_filename)
            # Check if path is a file
            if os.path.isfile(image_path):
                image = Image.open(image_path)
                # Define four different strength levels
                strength_levels = [0.1, 0.2, 0.3, 0.4]
                for strength_idx, strength in enumerate(strength_levels):
                    dynamic_transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ColorJitter(brightness=strength, contrast=strength, saturation=strength, hue=strength),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    transformed_image = dynamic_transform(image)
                    transformed_image = transforms.ToPILImage()(transformed_image)
                    save_path = os.path.join(target_class_dir, f"{os.path.splitext(image_filename)[0]}_strength{strength_idx+1}.jpg")
                    transformed_image.save(save_path)
            else:
                print(f"Skip directory: {image_path}")

def remove_hidden_folders(path):
    """Remove hidden folders from dataset path"""
    for item in os.listdir(path):
        if item.startswith('.'):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)

def stratified_split(dataset, train_size, val_size, test_size):
    """Stratified split of dataset into train, validation, and test sets"""
    X = [s[0] for s in dataset.samples]
    y = [s[1] for s in dataset.samples]
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_size+test_size, random_state=42)
    train_idx, temp_idx = next(sss1.split(X, y))
    temp_y = [y[i] for i in temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size/(val_size+test_size), random_state=42)
    val_idx, test_idx = next(sss2.split([X[i] for i in temp_idx], temp_y))
    val_idx = [temp_idx[i] for i in val_idx]
    test_idx = [temp_idx[i] for i in test_idx]
    return train_idx, val_idx, test_idx

def print_class_distribution(dataset, name):
    """Print class distribution in dataset"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)
    print(f"{name} class distribution: {class_counts}")

# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class MixedConv2d(nn.Module):
    """Mixed convolution with multiple kernel sizes"""
    def __init__(self, in_channels, out_channels=256, kernel_sizes=[1, 3, 5], stride=1, padding=1):
        super(MixedConv2d, self).__init__()
        # Adjust output channels for each convolution kernel
        split_channels = [out_channels // len(kernel_sizes) + (1 if i < out_channels % len(kernel_sizes) else 0) for i in range(len(kernel_sizes))]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, split_channels[i], kernel_size=kernel_sizes[i], stride=stride, padding=kernel_sizes[i]//2)
            for i in range(len(kernel_sizes))
        ])

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, 1)

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        # Ensure pointwise convolution output channels match depthwise convolution output channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        depthwise_out = self.depthwise(x)
        pointwise_out = self.pointwise(depthwise_out)
        # Ensure depthwise and pointwise convolution output channels are the same, assumed to be 256 here
        combined_out = depthwise_out * pointwise_out  # Element-wise multiplication
        return combined_out

class DynamicMultiHeadAttention(nn.Module):
    """Dynamic Multi-Head Attention mechanism"""
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(DynamicMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        
    def split_into_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / math.sqrt(self.depth)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        output = self.dense(output)
        
        return output
    
class ALGANet(nn.Module):
    """Adaptive Local-Global Attention Network"""
    def __init__(self, channels, enhance_local_features=True):
        super(ALGANet, self).__init__()
        self.enhance_local_features = enhance_local_features
        self.local_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)  # Modified to accept 256 channels
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(256 * 2, 256 // 2)  # Adjust as needed
        self.fc2 = nn.Linear(256 // 2, 256)  # Adjust as needed
        self.sigmoid = nn.Sigmoid()

        if self.enhance_local_features:
            self.enhance_conv = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, bias=False),  # Ensure channel count is 256
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),  # Ensure channel count is 256
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.enhance_local_features:
            x = self.enhance_conv(x)

        local_out = self.local_attention(x)
        global_out = self.global_attention(x)
        global_out = global_out.view(x.size(0), -1, 1, 1)
        global_out = global_out.expand_as(x)
        out = x * local_out * global_out
        return out

    def local_attention(self, x):
        local_feat = self.local_conv(x)
        local_out = self.sigmoid(local_feat)
        return local_out

    def global_attention(self, x):
        avg_pooled = self.global_avg_pool(x)
        max_pooled = self.global_max_pool(x)
        pooled_feat = torch.cat([avg_pooled, max_pooled], dim=1)
        pooled_feat = pooled_feat.view(x.size(0), -1)
        fc1_out = self.fc1(pooled_feat)
        fc2_out = self.fc2(fc1_out)
        global_out = self.sigmoid(fc2_out)
        return global_out.view(x.size(0), -1, 1, 1)

# ============================================================================
# MAIN MODEL ARCHITECTURE
# ============================================================================

class CapsNetWithALGA_DMA(nn.Module):
    """Cloud Classification Model with ALGA and Dynamic Multi-Head Attention"""
    def __init__(self, num_classes=11):
        super(CapsNetWithALGA_DMA, self).__init__()
        self.dense_net = densenet121(pretrained=True).features
        self.transition = nn.Conv2d(1024, 256, kernel_size=1)
        self.mixed_conv = MixedConv2d(256, 256, kernel_sizes=[1, 3, 5])
        self.depthwise_conv = DepthwiseSeparableConv2d(256, 256, kernel_size=3)
        self.alganet = ALGANet(256)  # Ensure ALGANet receives correct channel count
        self.channel_adjust = nn.Conv2d(256, 3, 1)  # Adjust from 256 channels to 3 channels
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Identity()  # Remove ViT original classification head
        self.dma = DynamicMultiHeadAttention(num_heads=8, d_model=768)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.dense_net(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (224, 224))
        x = self.transition(x)
        x = self.mixed_conv(x)
        x = self.depthwise_conv(x)
        x = self.alganet(x)  # Process through ALGANet first
        x = self.channel_adjust(x)  # Adjust channel count before passing to ViT
        x = self.vit(x)
        x = x.view(x.size(0), -1)
        x = self.dma(x, x, x, None)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # Remove extra dimension
        x = self.classifier(x)
        return x

# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def plot_metrics(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('train_val_metrics.png')
    plt.show()

def evaluate_model(model, test_loader):
    """Evaluate model performance on test set"""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    # Read class names
    class_names = test_loader.dataset.dataset.classes  # Ensure correct path

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)  # Compress outputs shape from [32, 1, 11] to [32, 11]

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())  # Save probabilities for all classes

    all_probabilities = np.array(all_probabilities)  # Convert list to numpy array for multi-dimensional indexing
    all_labels = np.array(all_labels)  # Ensure all_labels is numpy array

    # Calculate overall precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')

    # Calculate and log precision, recall, F1 score for each class
    precision_each, recall_each, f1_each, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
    
    for i, class_name in enumerate(class_names):
        logging.info(f'Class {class_name}: Precision: {precision_each[i]:.4f}, Recall: {recall_each[i]:.4f}, F1: {f1_each[i]:.4f}')

    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Calculate and plot PR curve for each class
    plt.figure(figsize=(12, 8))
    for i in range(len(class_names)):
        # Convert to binary classification problem for each class
        binary_labels = (all_labels == i).astype(int)
        class_probabilities = all_probabilities[:, i]
        
        precision_curve, recall_curve, _ = precision_recall_curve(binary_labels, class_probabilities)
        plt.plot(recall_curve, precision_curve, label=f'{class_names[i]}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend()
    plt.grid(True)
    plt.savefig('ppp.png')
    plt.show()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    
    logging.info(f'Test Accuracy: {100 * correct / total:.2f}%')
    logging.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    
    return {
        'accuracy': correct / total, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1,
        'confusion_matrix': cm,
        'class_metrics': {
            'precision': precision_each,
            'recall': recall_each,
            'f1': f1_each
        }
    }

def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=10):
    """Train the model"""
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        correct_train = 0
        total_train = 0

        # Wrap train_loader with tqdm for progress bar
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(1)  # Ensure output shape is [batch_size, num_classes]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        history['train_loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_accuracy)

        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze(1)  # Ensure output shape is [batch_size, num_classes]
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()

                _, predicted_val = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        history['val_loss'].append(total_val_loss / len(validation_loader))
        history['val_acc'].append(val_accuracy)
        
        logging.info(f'Epoch {epoch+1}: Training Loss {total_loss / len(train_loader):.4f}, Training Accuracy {train_accuracy:.2f}%')
        logging.info(f'Epoch {epoch+1}: Validation Loss {total_val_loss / len(validation_loader):.4f}, Validation Accuracy {val_accuracy:.2f}%')
        print(f'Epoch {epoch+1}: Training Loss {total_loss / len(train_loader):.4f}, Training Accuracy {train_accuracy:.2f}%, Validation Loss {total_val_loss / len(validation_loader):.4f}, Validation Accuracy {val_accuracy:.2f}%')

        # Save model
        torch.save(model.state_dict(), 'CapsNetViT_11_classes.pth')

        # Save training/validation curves
        plot_metrics(history)  # plot_metrics saves train_val_metrics.png

        # Evaluate and save confusion matrix, PR curves, etc.
        evaluate_model(model, test_loader)  # Saves confusion_matrix.png, ppp.png, etc.

    print('Training completed')
    return history

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# MAIN EXECUTION SECTION
# ============================================================================

if __name__ == "__main__":
    # Dataset paths
    dataset_path = r'GCD/fake'
    augmented_dataset_path = r'GCD/fake_augmented'

    # Remove hidden folders before loading ImageFolder
    if os.path.exists(augmented_dataset_path):
        remove_hidden_folders(augmented_dataset_path)

    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load dataset
    generate_augmented_images(dataset_path, augmented_dataset_path, num_variants=5)

    total_dataset = datasets.ImageFolder(augmented_dataset_path, transform=data_transforms['train'])
    train_idx, val_idx, test_idx = stratified_split(total_dataset, 0.7, 0.2, 0.1)
    from torch.utils.data import Subset
    train_dataset = Subset(total_dataset, train_idx)
    validation_dataset = Subset(total_dataset, val_idx)
    test_dataset = Subset(total_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model and deploy to GPU
    model = CapsNetWithALGA_DMA(num_classes=7).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=0.00001)

    # Output parameter count after model definition
    total_params = count_parameters(model)
    print(f'Total model parameters: {total_params}')
    logging.info(f'Total model parameters: {total_params}')

    # Print class distributions
    print_class_distribution(train_dataset, "Training set")
    print_class_distribution(validation_dataset, "Validation set")
    print_class_distribution(test_dataset, "Test set")

    # Start training process
    history = train_model(model, train_loader, validation_loader, criterion, optimizer)

    # Call evaluate_model to test the model
    evaluate_model(model, test_loader)

    # Save model
    torch.save(model.state_dict(), 'CapsNetViT_11_classes.pth')

    print("Training and evaluation completed successfully!")

# ============================================================================
# END OF COMBINED CODE FILE
# ============================================================================