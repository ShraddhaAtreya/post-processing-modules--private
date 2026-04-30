#!/bin/bash
# Quick Test - Run this after replacing checkpoint/mapping files

echo "=========================================="
echo "EXTRACTION PIPELINE - QUICK TEST"
echo "=========================================="
echo ""

echo "[1/4] Checking model files exist..."
if [ -f "models/hybrid_kannada_ocr_20251204_151641.pth" ]; then
    SIZE=$(du -h "models/hybrid_kannada_ocr_20251204_151641.pth" | cut -f1)
    echo "✅ Model file found ($SIZE)"
else
    echo "❌ Model file NOT found"
fi

if [ -f "models/class_mapping_20251204_151641.json" ]; then
    echo "✅ Class mapping found"
else
    echo "❌ Class mapping NOT found"
fi

echo ""
echo "[2/4] Testing model weights loading..."
.venv\Scripts\python.exe test_model.py 2>&1 | grep -E "(Missing keys|Unexpected keys|[0-9]+.[0-9]+%)" | head -5

echo ""
echo "[3/4] Checking class mapping..."
.venv\Scripts\python.exe -c "
import json
with open('models/class_mapping_20251204_151641.json') as f:
    mapping = json.load(f)
    classes = list(mapping['class_to_idx'].keys())
    print(f'First 5 classes: {classes[:5]}')
    # Check if they're Kannada or numeric
    if classes[0].startswith('0'):
        print('❌ Classes are NUMERIC (corrupted mapping)')
    else:
        print('✅ Classes are KANNADA (correct mapping)')
" 2>&1

echo ""
echo "[4/4] Checking model prediction confidence..."
.venv\Scripts\python.exe test_model.py 2>&1 | grep "Confidence:" 

echo ""
echo "=========================================="
