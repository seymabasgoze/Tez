"""
Diagnostic script: Check what the classification models actually output.
Run from PyQt_App directory:  python diagnose.py <image_path>
"""
import sys, os
import torch
import numpy as np
from inference import EnsemblePredictor

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python diagnose.py <fundus_görüntüsü_yolu>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"HATA: Dosya bulunamadı: {image_path}")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    efn_dir = os.path.join(base_dir, "efficientnet_b0")
    dense_dir = os.path.join(base_dir, "densenet121")
    unetpp_dir = os.path.join(base_dir, "unetpp")

    print("=" * 60)
    print("MODEL TANIMLAMA TESTİ")
    print("=" * 60)
    print(f"Görüntü: {image_path}")
    print()

    # Load predictor
    predictor = EnsemblePredictor(efn_dir, dense_dir, unetpp_dir, device="cpu")

    # Get vessel map
    vessel_soft, vessel_vis, orig_rgb = predictor.segmenter.segment(image_path)
    print(f"\nDamar haritası istatistikleri:")
    print(f"  min={vessel_soft.min():.4f}, max={vessel_soft.max():.4f}, mean={vessel_soft.mean():.4f}")

    # Prepare inputs
    from PIL import Image
    rgb_pil = Image.fromarray(orig_rgb)
    vessel_gray = (vessel_soft * 255).astype(np.uint8)
    vessel_pil = Image.fromarray(vessel_gray)

    rgb_t = predictor.rgb_transform(rgb_pil).unsqueeze(0)
    ves_t = predictor.vessel_transform(vessel_pil).unsqueeze(0)

    print(f"\nRGB tensor shape: {rgb_t.shape}, min={rgb_t.min():.3f}, max={rgb_t.max():.3f}")
    print(f"Vessel tensor shape: {ves_t.shape}, min={ves_t.min():.3f}, max={ves_t.max():.3f}")

    # Test each model individually
    print("\n" + "=" * 60)
    print("EfficientNet-B0 FOLDlar:")
    print("=" * 60)
    for i, m in enumerate(predictor.efn_models, 1):
        with torch.no_grad():
            logit = m(rgb_t, ves_t)
            prob = torch.sigmoid(logit).item()
            raw_logit = logit.item()
        print(f"  Fold {i}: logit={raw_logit:+.4f}  →  sigmoid={prob:.4f}  →  {'HASTA' if prob >= 0.5 else 'SAĞLIKLI'}")

    print("\n" + "=" * 60)
    print("DenseNet-121 FOLDlar:")
    print("=" * 60)
    for i, m in enumerate(predictor.dense_models, 1):
        with torch.no_grad():
            logit = m(rgb_t, ves_t)
            prob = torch.sigmoid(logit).item()
            raw_logit = logit.item()
        print(f"  Fold {i}: logit={raw_logit:+.4f}  →  sigmoid={prob:.4f}  →  {'HASTA' if prob >= 0.5 else 'SAĞLIKLI'}")

    # Ensemble results
    print("\n" + "=" * 60)
    print("ENSEMBLE SONUÇLARI:")
    print("=" * 60)

    efn_probs = []
    for m in predictor.efn_models:
        with torch.no_grad():
            efn_probs.append(torch.sigmoid(m(rgb_t, ves_t)).item())

    dense_probs = []
    for m in predictor.dense_models:
        with torch.no_grad():
            dense_probs.append(torch.sigmoid(m(rgb_t, ves_t)).item())

    p_efn = np.mean(efn_probs) if efn_probs else 0.0
    p_dense = np.mean(dense_probs) if dense_probs else 0.0
    p_final = (p_efn + p_dense) / 2

    print(f"  EfficientNet ortalama prob: {p_efn:.4f}")
    print(f"  DenseNet ortalama prob:     {p_dense:.4f}")
    print(f"  Final ensemble prob:        {p_final:.4f}")
    print()
    print(f"  SONUÇ (threshold=0.5): {'⚠️  HASTA' if p_final >= 0.5 else '✅ SAĞLIKLI'}")
    print()
    print("  Eğer label ters ise (1=sağlıklı, 0=hasta):")
    print(f"  SONUÇ (ters label):   {'⚠️  HASTA' if p_final < 0.5 else '✅ SAĞLIKLI'}")

    # Check model weight statistics
    print("\n" + "=" * 60)
    print("MODEL HEAD AĞIRLIK İSTATİSTİKLERİ:")
    print("=" * 60)
    
    if predictor.efn_models:
        m = predictor.efn_models[0]
        head_w = m.head[1].weight.data
        head_b = m.head[1].bias.data
        print(f"  EfficientNet fold1 head.weight: shape={head_w.shape}, mean={head_w.mean():.6f}, std={head_w.std():.6f}")
        print(f"  EfficientNet fold1 head.bias:   {head_b.item():.6f}")
    
    if predictor.dense_models:
        m = predictor.dense_models[0]
        head_w = m.head[1].weight.data
        head_b = m.head[1].bias.data
        print(f"  DenseNet fold1 head.weight:     shape={head_w.shape}, mean={head_w.mean():.6f}, std={head_w.std():.6f}")
        print(f"  DenseNet fold1 head.bias:       {head_b.item():.6f}")

if __name__ == "__main__":
    main()
