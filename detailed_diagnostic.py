"""
Detailed diagnostic script to identify the exact checkpoint issue
"""

import torch
import json
import os

print("=" * 80)
print("KANNADA OCR - DETAILED DIAGNOSTIC")
print("=" * 80)

# Check if files exist
print("\n[1] FILE EXISTENCE CHECK:")
files_to_check = [
    "models/hybrid_kannada_ocr_20251204_151641.pth",
    "models/class_mapping_20251204_151641.json",
    "models/model_weights_20251204_151641.pth"
]

for filepath in files_to_check:
    exists = os.path.exists(filepath)
    size = os.path.getsize(filepath) if exists else 0
    status = "✅" if exists else "❌"
    print(f"   {status} {filepath:<60} ({size:,} bytes)")

# Load and inspect checkpoint
print("\n[2] CHECKPOINT STATE DICT INSPECTION:")
print("   Loading: models/hybrid_kannada_ocr_20251204_151641.pth")

try:
    ckpt = torch.load("models/hybrid_kannada_ocr_20251204_151641.pth", map_location='cpu')
    
    # Check structure
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            print(f"   ✅ Found 'model_state_dict' key (wrapped checkpoint)")
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print(f"   ✅ Found 'state_dict' key")
        else:
            state_dict = ckpt
            print(f"   ⚠️  Direct state dict (no wrapper)")
    else:
        state_dict = ckpt
        print(f"   ⚠️  Direct tensor/state dict")
    
    # Analyze keys
    keys = list(state_dict.keys())
    print(f"\n   State dict has {len(keys)} keys")
    
    # Show first 10 keys
    print(f"\n   First 10 keys:")
    for i, key in enumerate(keys[:10], 1):
        param = state_dict[key]
        shape = param.shape if hasattr(param, 'shape') else 'N/A'
        print(f"      {i:2d}. {key:<70} {shape}")
    
    # Check for key name patterns
    print(f"\n   Key naming patterns:")
    features_count = sum(1 for k in keys if k.startswith('features.'))
    densenet_features_count = sum(1 for k in keys if k.startswith('densenet_features.'))
    other_count = len(keys) - features_count - densenet_features_count
    
    print(f"      Keys starting with 'features.':           {features_count}")
    print(f"      Keys starting with 'densenet_features.':  {densenet_features_count}")
    print(f"      Other keys:                               {other_count}")
    
    if features_count > 0 and densenet_features_count == 0:
        print(f"\n   ⚠️  ISSUE FOUND: Keys use 'features.' not 'densenet_features.'")
        print(f"      Need to rename all 'features.' → 'densenet_features.'")
        print(f"      This is why DenseNet weights aren't loading!")
    elif densenet_features_count > 0:
        print(f"\n   ✅ Keys use correct 'densenet_features.' naming")
    
except Exception as e:
    print(f"   ❌ Error loading checkpoint: {e}")

# Load and inspect class mapping
print("\n[3] CLASS MAPPING INSPECTION:")
print("   Loading: models/class_mapping_20251204_151641.json")

try:
    with open("models/class_mapping_20251204_151641.json") as f:
        mapping_data = json.load(f)
    
    if 'num_classes' in mapping_data:
        num_classes = mapping_data['num_classes']
        print(f"   ✅ Found 'num_classes': {num_classes}")
    
    if 'class_to_idx' in mapping_data:
        class_to_idx = mapping_data['class_to_idx']
        print(f"   ✅ Found 'class_to_idx' with {len(class_to_idx)} entries")
        
        # Show first 10 classes
        classes = list(class_to_idx.keys())[:10]
        print(f"\n   First 10 classes:")
        for i, cls in enumerate(classes, 1):
            idx = class_to_idx[cls]
            is_kannada = ord(cls[0]) >= 0x0C80 if len(cls) > 0 else False
            if is_kannada:
                print(f"      {i:2d}. '{cls}' (U+{ord(cls[0]):04X}) → idx {idx}")
            else:
                print(f"      {i:2d}. '{cls}' → idx {idx}  ⚠️  NOT KANNADA")
        
        # Check if all are numeric
        first_class = classes[0] if classes else ""
        if first_class and first_class[0].isdigit():
            print(f"\n   ❌ ISSUE FOUND: Classes are NUMERIC ('{first_class}', '{classes[1]}', ...)")
            print(f"      Should be KANNADA characters like 'ಅ', 'ಆ', 'ಇ'")
            print(f"      Mapping file is corrupted or placeholder!")
        elif all(ord(c[0]) >= 0x0C80 for c in classes if c):
            print(f"\n   ✅ Classes are KANNADA characters (correct)")
        else:
            print(f"\n   ⚠️  Mixed content in classes")
    
except Exception as e:
    print(f"   ❌ Error loading mapping: {e}")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY:")
print("=" * 80)

print("""
The zero-character extraction is caused by:

1. ❌ Model weights mismatch
   - Checkpoint has 'features.*' keys
   - Model expects 'densenet_features.*' keys  
   - DenseNet weights NOT loaded → random predictions

2. ❌ Corrupted class mapping
   - Classes are numeric ('0000', '0001', etc.)
   - Should be Kannada characters ('ಅ', 'ಆ', etc.)
   - Output is wrong even if model worked

SOLUTION:
- Find correct checkpoint with 'densenet_features.*' keys
- Find correct mapping with Kannada characters
- See ROOT_CAUSE_ANALYSIS.md for detailed recovery steps
""")

print("=" * 80)
