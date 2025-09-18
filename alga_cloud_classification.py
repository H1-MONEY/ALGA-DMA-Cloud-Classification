# ALGA Cloud Classification Model
# A deep learning model for cloud classification using ALGA (Adaptive Local-Global Attention) mechanism
# Combined with DenseNet, Vision Transformer, and Dynamic Multi-Head Attention

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import shutil
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

logging.basicConfig(filename='training_log.log', level=logging.INFO)

# Set GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset paths
dataset_path = r'GCD/fake'

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

# Data transforms - 修正为实时增强
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 添加实时颜色抖动
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

class TransformSubset(Subset):
    """带变换的子集类"""
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        if self.transform:
            image = self.transform(image)
        return image, label

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
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        depthwise_out = self.depthwise(x)
        pointwise_out = self.pointwise(depthwise_out)
        combined_out = depthwise_out * pointwise_out
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
        self.local_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(256 * 2, 256 // 2)
        self.fc2 = nn.Linear(256 // 2, 256)
        self.sigmoid = nn.Sigmoid()

        if self.enhance_local_features:
            self.enhance_conv = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
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

# 修正后的模型类名
class ALGA_DenseNet(nn.Module):
    """ALGA-DenseNet: Cloud Classification Model with ALGA and Dynamic Multi-Head Attention"""
    def __init__(self, num_classes=7):
        super(ALGA_DenseNet, self).__init__()
        self.dense_net = densenet121(pretrained=True).features
        self.transition = nn.Conv2d(1024, 256, kernel_size=1)
        self.mixed_conv = MixedConv2d(256, 256, kernel_sizes=[1, 3, 5])
        self.depthwise_conv = DepthwiseSeparableConv2d(256, 256, kernel_size=3)
        self.alganet = ALGANet(256)
        self.channel_adjust = nn.Conv2d(256, 3, 1)
        
        # 修正后的ViT集成
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # 保留ViT的编码器部分，移除分类头
        self.vit.heads.head = nn.Identity()
        
        self.dma = DynamicMultiHeadAttention(num_heads=8, d_model=768)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # DenseNet特征提取
        x = self.dense_net(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (224, 224))
        x = self.transition(x)
        
        # 混合卷积和深度可分离卷积
        x = self.mixed_conv(x)
        x = self.depthwise_conv(x)
        
        # ALGA注意力机制
        x = self.alganet(x)
        
        # 调整通道数以适配ViT
        x = self.channel_adjust(x)
        
        # 修正后的ViT处理 - 获取序列特征而不是分类特征
        # 使用ViT的内部处理来获取patch embeddings序列
        x = self.vit._process_input(x)
        n = x.shape[0]
        
        # 添加class token
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # 添加位置编码
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        
        # 通过Transformer编码器层
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)
        
        # 现在x是序列数据，形状为[batch_size, num_patches + 1, 768]
        # 适合输入到DMA
        x = self.dma(x, x, x, None)
        
        # 取出class token用于分类
        x = x[:, 0, :]  # 选择第一个token (class token)
        
        x = self.classifier(x)
        return x

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

    class_names = test_loader.dataset.dataset.classes

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    precision_each, recall_each, f1_each, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
    
    for i, class_name in enumerate(class_names):
        logging.info(f'Class {class_name}: Precision: {precision_each[i]:.4f}, Recall: {recall_each[i]:.4f}, F1: {f1_each[i]:.4f}')

    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    plt.figure(figsize=(12, 8))
    for i in range(len(class_names)):
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
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
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

        torch.save(model.state_dict(), 'ALGA_DenseNet_model.pth')

    print('Training completed')
    return history

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 移除隐藏文件夹
    if os.path.exists(dataset_path):
        remove_hidden_folders(dataset_path)

    # 直接使用原始数据集，不进行预生成增强
    total_dataset = datasets.ImageFolder(dataset_path, transform=None)
    train_idx, val_idx, test_idx = stratified_split(total_dataset, 0.7, 0.2, 0.1)

    # 创建带变换的子集
    train_dataset = TransformSubset(total_dataset, train_idx, transform=data_transforms['train'])
    validation_dataset = TransformSubset(total_dataset, val_idx, transform=data_transforms['val'])
    test_dataset = TransformSubset(total_dataset, test_idx, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 使用修正后的模型类名
    model = ALGA_DenseNet(num_classes=7).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=0.00001)

    total_params = count_parameters(model)
    print(f'Total model parameters: {total_params}')
    logging.info(f'Total model parameters: {total_params}')

    print_class_distribution(train_dataset, "Training set")
    print_class_distribution(validation_dataset, "Validation set")
    print_class_distribution(test_dataset, "Test set")

    history = train_model(model, train_loader, validation_loader, criterion, optimizer)
    evaluate_model(model, test_loader)
    torch.save(model.state_dict(), 'ALGA_DenseNet_model.pth')

    print("Training and evaluation completed successfully!")