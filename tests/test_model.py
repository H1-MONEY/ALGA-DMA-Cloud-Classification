"""
Unit tests for ALGA-DMA model components
"""

import unittest
import torch
import torch.nn as nn
from src.model import (
    MixedConv2d,
    DepthwiseSeparableConv2d,
    DynamicMultiHeadAttention,
    ALGANet,
    CapsNetWithALGA_DMA
)


class TestModelComponents(unittest.TestCase):
    """Test individual model components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.channels = 64
        self.height = 32
        self.width = 32
        
    def test_mixed_conv2d(self):
        """Test MixedConv2d layer"""
        layer = MixedConv2d(
            in_channels=32,
            out_channels=64,
            kernel_sizes=[3, 5, 7],
            stride=1,
            padding='same'
        ).to(self.device)
        
        x = torch.randn(self.batch_size, 32, self.height, self.width).to(self.device)
        output = layer(x)
        
        # Check output shape
        expected_shape = (self.batch_size, 64, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_depthwise_separable_conv2d(self):
        """Test DepthwiseSeparableConv2d layer"""
        layer = DepthwiseSeparableConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        ).to(self.device)
        
        x = torch.randn(self.batch_size, 32, self.height, self.width).to(self.device)
        output = layer(x)
        
        # Check output shape
        expected_shape = (self.batch_size, 64, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_dynamic_multi_head_attention(self):
        """Test DynamicMultiHeadAttention layer"""
        embed_dim = 128
        num_heads = 8
        
        layer = DynamicMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        ).to(self.device)
        
        # Create input tensor (batch_size, seq_len, embed_dim)
        seq_len = 16
        x = torch.randn(self.batch_size, seq_len, embed_dim).to(self.device)
        output = layer(x)
        
        # Check output shape
        expected_shape = (self.batch_size, seq_len, embed_dim)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_alga_net(self):
        """Test ALGANet component"""
        layer = ALGANet(
            in_channels=64,
            out_channels=128,
            num_heads=8
        ).to(self.device)
        
        x = torch.randn(self.batch_size, 64, self.height, self.width).to(self.device)
        output = layer(x)
        
        # Check output shape
        expected_shape = (self.batch_size, 128, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestFullModel(unittest.TestCase):
    """Test the complete CapsNetWithALGA_DMA model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.num_classes = 10
        self.input_size = (3, 224, 224)  # RGB images
        
    def test_model_forward_pass(self):
        """Test forward pass of the complete model"""
        model = CapsNetWithALGA_DMA(
            num_classes=self.num_classes,
            input_channels=3
        ).to(self.device)
        
        # Create random input
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Check output is properly normalized (softmax)
        probabilities = torch.softmax(output, dim=1)
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(self.batch_size).to(self.device)))
        
    def test_model_backward_pass(self):
        """Test backward pass and gradient computation"""
        model = CapsNetWithALGA_DMA(
            num_classes=self.num_classes,
            input_channels=3
        ).to(self.device)
        
        # Create random input and target
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        target = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}")
                
    def test_model_different_input_sizes(self):
        """Test model with different input sizes"""
        model = CapsNetWithALGA_DMA(
            num_classes=self.num_classes,
            input_channels=3
        ).to(self.device)
        
        # Test different input sizes
        input_sizes = [(3, 128, 128), (3, 256, 256), (3, 224, 224)]
        
        for input_size in input_sizes:
            with self.subTest(input_size=input_size):
                x = torch.randn(self.batch_size, *input_size).to(self.device)
                output = model(x)
                
                # Check output shape
                expected_shape = (self.batch_size, self.num_classes)
                self.assertEqual(output.shape, expected_shape)
                
    def test_model_parameter_count(self):
        """Test that model has reasonable number of parameters"""
        model = CapsNetWithALGA_DMA(
            num_classes=self.num_classes,
            input_channels=3
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Check that model has parameters
        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)
        
        # Check that all parameters are trainable by default
        self.assertEqual(total_params, trainable_params)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


class TestModelUtils(unittest.TestCase):
    """Test utility functions for the model"""
    
    def test_model_summary(self):
        """Test model summary functionality"""
        from src.model import count_parameters
        
        model = CapsNetWithALGA_DMA(num_classes=10, input_channels=3)
        param_count = count_parameters(model)
        
        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)
        
    def test_model_device_compatibility(self):
        """Test model works on both CPU and GPU"""
        model = CapsNetWithALGA_DMA(num_classes=10, input_channels=3)
        
        # Test on CPU
        x_cpu = torch.randn(1, 3, 224, 224)
        output_cpu = model(x_cpu)
        self.assertEqual(output_cpu.device.type, 'cpu')
        
        # Test on GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            x_gpu = x_cpu.cuda()
            output_gpu = model_gpu(x_gpu)
            self.assertEqual(output_gpu.device.type, 'cuda')


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run tests
    unittest.main(verbosity=2)