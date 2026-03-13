"""
Deep diagnostic: inspect checkpoint contents vs model architecture.
Run: python diagnose2.py
"""
import os, sys, torch
import torchvision.models as tv_models
import torch.nn as nn
import timm

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
efn_dir = os.path.join(base_dir, "efficientnet_b0")
dense_dir = os.path.join(base_dir, "densenet121")

def inspect_checkpoint(path, label):
    print(f"\n{'='*70}")
    print(f"  {label}: {os.path.basename(path)}")
    print(f"{'='*70}")
    
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    # Check if it's a dict with 'model'/'state_dict' key or direct state dict
    if isinstance(ckpt, dict):
        print(f"  Checkpoint type: dict with keys = {list(ckpt.keys())}")
        if "model" in ckpt:
            state = ckpt["model"]
            print("  Using ckpt['model']")
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
            print("  Using ckpt['state_dict']")
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            print("  Using ckpt['model_state_dict']")
        else:
            # Check if keys look like state dict keys
            first_key = list(ckpt.keys())[0]
            if '.' in first_key or 'weight' in first_key:
                state = ckpt
                print("  Direct state dict (no wrapper)")
            else:
                print(f"  UYARI: Bilinmeyen checkpoint yapısı!")
                for k, v in ckpt.items():
                    if isinstance(v, dict):
                        print(f"    '{k}': dict with {len(v)} keys")
                    elif isinstance(v, torch.Tensor):
                        print(f"    '{k}': Tensor {v.shape}")
                    else:
                        print(f"    '{k}': {type(v).__name__} = {v}")
                return None
    else:
        state = ckpt
        print(f"  Checkpoint type: {type(ckpt).__name__}")
    
    print(f"\n  Toplam parametre sayısı: {len(state)}")
    
    # Show all keys grouped by prefix
    prefixes = {}
    for k in state.keys():
        prefix = k.split('.')[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(k)
    
    print(f"\n  Üst seviye prefix'ler:")
    for prefix, keys in prefixes.items():
        print(f"    {prefix}: {len(keys)} parametre")
    
    # Show specific important layers
    print(f"\n  Head/Classifier katmanları:")
    for k, v in state.items():
        if any(x in k.lower() for x in ['head', 'classifier', 'fc', 'linear']):
            print(f"    {k}: shape={v.shape}")
    
    # Show first and last 5 keys
    keys = list(state.keys())
    print(f"\n  İlk 5 anahtar:")
    for k in keys[:5]:
        print(f"    {k}: {state[k].shape}")
    print(f"\n  Son 5 anahtar:")
    for k in keys[-5:]:
        print(f"    {k}: {state[k].shape}")
    
    return state

def show_model_keys(model, label):
    print(f"\n{'='*70}")
    print(f"  BEKLENen mimari: {label}")
    print(f"{'='*70}")
    state = model.state_dict()
    prefixes = {}
    for k in state.keys():
        prefix = k.split('.')[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(k)
    
    print(f"  Toplam parametre: {len(state)}")
    print(f"  Üst seviye prefix'ler:")
    for prefix, keys in prefixes.items():
        print(f"    {prefix}: {len(keys)} parametre")
    
    # Show head
    print(f"\n  Head/Classifier katmanları:")
    for k, v in state.items():
        if any(x in k.lower() for x in ['head', 'classifier', 'fc', 'linear']):
            print(f"    {k}: {v.shape}")
    
    keys = list(state.keys())
    print(f"\n  İlk 5 anahtar:")
    for k in keys[:5]:
        print(f"    {k}: {state[k].shape}")
    print(f"\n  Son 5 anahtar:")
    for k in keys[-5:]:
        print(f"    {k}: {state[k].shape}")

def key_comparison(ckpt_state, model_state, label):
    print(f"\n{'='*70}")
    print(f"  ANAHTAR KARŞILAŞTIRMA: {label}")
    print(f"{'='*70}")
    ckpt_keys = set(ckpt_state.keys())
    model_keys = set(model_state.keys())
    
    missing_in_ckpt = model_keys - ckpt_keys
    extra_in_ckpt = ckpt_keys - model_keys
    common = ckpt_keys & model_keys
    
    print(f"  Ortak anahtarlar: {len(common)}")
    print(f"  Model'de var, checkpoint'ta yok: {len(missing_in_ckpt)}")
    print(f"  Checkpoint'ta var, model'de yok: {len(extra_in_ckpt)}")
    
    if missing_in_ckpt:
        print(f"\n  EKSIK (checkpoint'ta yok):")
        for k in sorted(missing_in_ckpt)[:10]:
            print(f"    {k}")
    
    if extra_in_ckpt:
        print(f"\n  FAZLA (model'de yok):")
        for k in sorted(extra_in_ckpt)[:10]:
            print(f"    {k}")
    
    # Check shape mismatches
    shape_mismatches = []
    for k in common:
        if ckpt_state[k].shape != model_state[k].shape:
            shape_mismatches.append((k, ckpt_state[k].shape, model_state[k].shape))
    
    if shape_mismatches:
        print(f"\n  BOYUT UYUMSUZLUKLARI:")
        for k, cs, ms in shape_mismatches:
            print(f"    {k}: ckpt={cs} vs model={ms}")
    else:
        print(f"\n  ✅ Ortak anahtarların boyutları uyumlu.")

# ===================== MAIN =====================
print("=" * 70)
print("  DERİN MODEL TEŞHİSİ")
print("=" * 70)

# 1. Inspect checkpoints
efn_ckpt = inspect_checkpoint(os.path.join(efn_dir, "fold1_best.pth"), "EfficientNet Fold1")
dense_ckpt = inspect_checkpoint(os.path.join(dense_dir, "fold1_best.pth"), "DenseNet Fold1")

# 2. Show expected model architectures
from inference import DualEfficientNetB0, DualDenseNet121

efn_model = DualEfficientNetB0()
dense_model = DualDenseNet121()

show_model_keys(efn_model, "DualEfficientNetB0")
show_model_keys(dense_model, "DualDenseNet121")

# 3. Compare keys
if efn_ckpt is not None:
    key_comparison(efn_ckpt, efn_model.state_dict(), "EfficientNet")

if dense_ckpt is not None:
    key_comparison(dense_ckpt, dense_model.state_dict(), "DenseNet")

print("\n\n✅ Teşhis tamamlandı.")
