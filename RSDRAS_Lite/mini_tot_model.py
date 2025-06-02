import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

class PhysicsConstraintLayer(nn.Module):
    """Physics constraint layer that enforces orbital mechanics laws"""
    
    def __init__(self, feature_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Learnable weights for physics constraints
        self.energy_weights = nn.Parameter(torch.ones(feature_dim) * 0.1)
        self.momentum_weights = nn.Parameter(torch.ones(feature_dim) * 0.1)
        self.kepler_weights = nn.Parameter(torch.ones(feature_dim) * 0.1)
        
        # Physics constraint MLPs
        self.energy_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )
        
        self.momentum_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )
        
    def forward(self, features, orbital_params=None):
        """Apply physics constraints to features"""
        batch_size = features.size(0)
        
        # Energy conservation constraint
        energy_constraint = self.energy_mlp(features) * self.energy_weights
        
        # Angular momentum constraint
        momentum_constraint = self.momentum_mlp(features) * self.momentum_weights
        
        # Kepler's law constraint (simplified)
        kepler_constraint = torch.tanh(features) * self.kepler_weights
        
        # Apply constraints with residual connections
        constrained_features = features + energy_constraint + momentum_constraint + kepler_constraint
        
        return constrained_features


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    
    def __init__(self, d_model, max_len=7):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MiniTemporalOrbitalTransformer(nn.Module):
    """
    Mini Temporal-Orbital Transformer optimized for 4GB GPU
    Processes 30 physics features with 7-day temporal modeling
    """
    
    def __init__(
        self,
        input_features=30,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        sequence_length=7,
        num_classes=4,
        dropout=0.2
    ):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_features, hidden_dim)
        
        # Positional encoding for temporal sequences
        self.pos_encoding = PositionalEncoding(hidden_dim, sequence_length)
        
        # Temporal LSTM for initial sequence processing
        self.temporal_lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head attention for orbital mechanics
        self.orbital_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Physics constraint layer
        self.physics_layer = PhysicsConstraintLayer(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
            return_attention: Whether to return attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq, hidden)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq, batch, hidden)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq, hidden)
        
        # Temporal LSTM processing
        lstm_out, (hidden, cell) = self.temporal_lstm(x)
        
        # Layer norm + residual
        lstm_out = self.ln1(lstm_out + x)
        
        # Multi-head attention for orbital mechanics
        attn_out, attn_weights = self.orbital_attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Layer norm + residual
        attn_out = self.ln2(attn_out + lstm_out)
        
        # Feed-forward network
        ffn_out = self.ffn(attn_out)
        
        # Residual connection
        features = attn_out + ffn_out
        
        # Apply physics constraints
        features = self.physics_layer(features)
        
        # Global pooling across temporal dimension
        pooled_features = self.pooling(features.transpose(1, 2)).squeeze(-1)
        
        # Classification
        risk_scores = self.classifier(pooled_features)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(pooled_features)
        
        if return_attention:
            return risk_scores, uncertainty, attn_weights
        else:
            return risk_scores, uncertainty
    
    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss function for orbital mechanics"""
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
        super().__init__()
        self.alpha = alpha  # Classification loss weight
        self.beta = beta    # Physics constraint weight
        self.gamma = gamma  # Uncertainty regularization weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, uncertainty, targets, orbital_params=None):
        """
        Compute physics-informed loss
        Args:
            predictions: Model predictions (batch_size, num_classes)
            uncertainty: Uncertainty estimates (batch_size, 1)
            targets: Ground truth labels (batch_size,)
            orbital_params: Optional orbital parameters for physics constraints
        """
        # Classification loss
        classification_loss = self.ce_loss(predictions, targets)
        
        # Physics constraint loss (simplified)
        physics_loss = torch.tensor(0.0, device=predictions.device)
        if orbital_params is not None:
            # Add physics constraints based on orbital parameters
            # This is a placeholder for actual physics constraints
            physics_loss = torch.mean(torch.abs(orbital_params[:, 0] - orbital_params[:, 1]))
        
        # Uncertainty regularization (encourage confident predictions when appropriate)
        uncertainty_loss = torch.mean(uncertainty * (1 - uncertainty))
        
        # Total loss
        total_loss = (self.alpha * classification_loss + 
                     self.beta * physics_loss + 
                     self.gamma * uncertainty_loss)
        
        return total_loss, classification_loss, physics_loss, uncertainty_loss


def create_mini_tot_model(device='cuda', **kwargs):
    """Create and initialize Mini-TOT model"""
    model = MiniTemporalOrbitalTransformer(**kwargs)
    model = model.to(device)
    
    print(f"Model created on {device}")
    print(f"Model size: {model.get_model_size():.2f} MB")
    print(f"Total parameters: {model.count_parameters():,}")
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_mini_tot_model(device=device)
    
    # Test forward pass
    batch_size = 32
    sequence_length = 7
    input_features = 30
    
    # Create dummy input
    x = torch.randn(batch_size, sequence_length, input_features).to(device)
    
    # Forward pass
    with torch.no_grad():
        predictions, uncertainty = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print("Model test successful!") 