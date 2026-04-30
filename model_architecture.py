"""
Model Architecture for Kannada OCR
DenseNet121 + DeiT + Self-Attention Hybrid
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
import torch.backends.cudnn as cudnn

# DEVICE DETECTION (explicit, no silent fallback)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
    cudnn.benchmark = True
    print(f"Using GPU: {GPU_NAME}")
else:
    DEVICE = torch.device('cpu')
    GPU_NAME = None
    print("Using CPU: CUDA not available")


class SelfAttention(nn.Module):
    """Self-Attention mechanism for spatial feature enhancement"""
    
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Query, Key, Value projections
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Attention scores
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable weight
        return self.gamma * out + x


class DenseNetDeiTHybrid(nn.Module):
    """
    Hybrid Architecture: DenseNet121 + DeiT + Self-Attention
    
    Components:
    1. DenseNet121: CNN backbone for local feature extraction
    2. Self-Attention: Spatial attention mechanism
    3. DeiT: Transformer for global context
    4. Fusion Layer: Combines features for classification
    """
    
    def __init__(self, num_classes=621, pretrained=True):
        super(DenseNetDeiTHybrid, self).__init__()
        
        # DenseNet121 branch
        densenet = models.densenet121(pretrained=pretrained)
        self.densenet_features = densenet.features  # Output: 1024 features
        densenet_out = 1024
        
        # Self-attention on DenseNet features
        self.attention = SelfAttention(densenet_out)
        
        # DeiT branch (small variant)
        self.deit = timm.create_model(
            'deit_small_patch16_224', 
            pretrained=pretrained, 
            num_classes=0  # Remove classification head
        )
        deit_out = 384
        
        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(densenet_out + deit_out, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # DenseNet path
        dense_feat = self.densenet_features(x)
        dense_feat = self.attention(dense_feat)
        dense_feat = nn.functional.adaptive_avg_pool2d(dense_feat, (1, 1))
        dense_feat = torch.flatten(dense_feat, 1)
        
        # DeiT path
        deit_feat = self.deit(x)
        
        # Fusion
        combined = torch.cat([dense_feat, deit_feat], dim=1)
        output = self.fusion(combined)
        
        return output


def load_model(model_path, class_mapping_path, device=None):
    """
    Load trained model with weights
    """
    import json
    # Resolve device
    if device is None:
        device = DEVICE
    else:
        device = torch.device(device if isinstance(device, str) else device)

    # Load class mapping
    with open(class_mapping_path, 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    
    num_classes = class_mapping['num_classes']
    idx_to_class = class_mapping['idx_to_class']
    
    print(f"✅ Loaded class mapping: {num_classes} classes")
    
    # Initialize model
    model = DenseNetDeiTHybrid(num_classes=num_classes, pretrained=False)

    # Load checkpoint (map to device for safe loading)
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception:
        # fallback to CPU load then move
        checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # FIX KEY MISMATCH: Rename 'features.' to 'densenet_features.'
    new_state_dict = {}
    renamed_count = 0
    
    for key, value in state_dict.items():
        if key.startswith('features.'):
            new_key = key.replace('features.', 'densenet_features.', 1)
            new_state_dict[new_key] = value
            renamed_count += 1
        else:
            new_state_dict[key] = value
    
    if renamed_count > 0:
        print(f"🔧 Renamed {renamed_count} keys: 'features.*' → 'densenet_features.*'")
    
    # Load corrected state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
    
    model.to(device)
    model.eval()
    # Disable grad globally for inference
    torch.set_grad_enabled(False)
    
    print(f"✅ Model loaded successfully")
    
    return model, idx_to_class


if __name__ == "__main__":
    # Test model creation
    print("Testing model architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DenseNetDeiTHybrid(num_classes=621, pretrained=False)
    model.to(device)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"✅ Model created successfully!")
    print(f"   Device: {device}")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")