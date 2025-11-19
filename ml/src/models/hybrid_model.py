"""
Hybrid Deep Learning Model for Network Intrusion Detection.

This module implements a neural network with attention mechanism for
both binary and multi-class classification of network attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionLayer(nn.Module):
    """
    Self-attention mechanism for feature importance weighting.
    """
    
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Calculate attention scores
        attention_scores = self.attention_weights(x)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention
        attended = x * attention_weights
        
        return attended, attention_weights


class HybridNIDSModel(nn.Module):
    """
    Hybrid deep neural network for network intrusion detection.
    
    Architecture:
    - Input layer with dropout
    - Multiple hidden layers with BatchNorm and Dropout
    - Attention mechanism
    - Separate heads for binary and multi-class classification
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.3,
        use_attention: bool = True
    ):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_attention: Whether to use attention mechanism
        """
        super(HybridNIDSModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_attention = use_attention
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
        
        # Attention layer
        if self.use_attention:
            self.attention = AttentionLayer(hidden_dims[-1])
        
        # Output heads
        self.binary_head = nn.Linear(hidden_dims[-1], 2)  # Binary classification
        self.multiclass_head = nn.Linear(hidden_dims[-1], 5)  # 5-class classification
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (binary_logits, multiclass_logits)
            If return_attention=True, returns (binary_logits, multiclass_logits, attention_weights)
        """
        # Input layer
        x = self.input_layer(x)
        
        # Hidden layers
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        
        # Attention
        attention_weights = None
        if self.use_attention:
            x, attention_weights = self.attention(x)
        
        # Output heads
        binary_logits = self.binary_head(x)
        multiclass_logits = self.multiclass_head(x)
        
        if return_attention and attention_weights is not None:
            return binary_logits, multiclass_logits, attention_weights
        
        return binary_logits, multiclass_logits


class BinaryNIDSModel(nn.Module):
    """
    Simplified model for binary classification only (normal vs attack).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.3
    ):
        super(BinaryNIDSModel, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 2))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MulticlassNIDSModel(nn.Module):
    """
    Simplified model for multi-class classification (normal, dos, probe, r2l, u2r).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.3
    ):
        super(MulticlassNIDSModel, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 5))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_model(
    model_type: str,
    input_dim: int,
    hidden_dims: list = [256, 128, 64],
    dropout_rate: float = 0.3,
    use_attention: bool = True
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: One of 'hybrid', 'binary', 'multiclass'
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability
        use_attention: Whether to use attention (hybrid model only)
        
    Returns:
        PyTorch model
    """
    if model_type == 'hybrid':
        return HybridNIDSModel(input_dim, hidden_dims, dropout_rate, use_attention)
    elif model_type == 'binary':
        return BinaryNIDSModel(input_dim, hidden_dims, dropout_rate)
    elif model_type == 'multiclass':
        return MulticlassNIDSModel(input_dim, hidden_dims, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing NIDS Models")
    print("=" * 60)
    
    input_dim = 122  # After one-hot encoding (example)
    
    # Test Hybrid Model
    print("\n1. Hybrid Model (Binary + Multiclass with Attention):")
    hybrid_model = create_model('hybrid', input_dim)
    print(f"   Parameters: {count_parameters(hybrid_model):,}")
    
    # Test input
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    binary_out, multiclass_out = hybrid_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Binary output shape: {binary_out.shape}")
    print(f"   Multiclass output shape: {multiclass_out.shape}")
    
    # Test Binary Model
    print("\n2. Binary Model:")
    binary_model = create_model('binary', input_dim)
    print(f"   Parameters: {count_parameters(binary_model):,}")
    binary_out = binary_model(x)
    print(f"   Output shape: {binary_out.shape}")
    
    # Test Multiclass Model
    print("\n3. Multiclass Model:")
    multiclass_model = create_model('multiclass', input_dim)
    print(f"   Parameters: {count_parameters(multiclass_model):,}")
    multiclass_out = multiclass_model(x)
    print(f"   Output shape: {multiclass_out.shape}")
    
    print("\n✓ All models created and tested successfully!")







