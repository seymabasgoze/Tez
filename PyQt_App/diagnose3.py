"""
Preprocessing brute-force test: Try multiple normalization & resize combinations
to find which one produces discriminative model outputs.
Run: python diagnose3.py <hasta_img> <saglikli_img>
"""
import sys, os, torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import DualEfficientNetB0, DualDenseNet121, VesselSegmenter

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
efn_dir = os.path.join(base_dir, "efficientnet_b0")
dense_dir = os.path.join(base_dir, "densenet121")
unetpp_dir = os.path.join(base_dir, "unetpp")

def load_one_model(folder, arch_type):
    path = os.path.join(folder, "fold1_best.pth")
    model = DualEfficientNetB0() if arch_type == "efficientnet" else DualDenseNet121()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def try_preprocessing(img_path, segmenter, efn_model, dense_model, preproc_name, rgb_transform, vessel_transform, img_size=224):
    """Test a specific preprocessing and return logits."""
    vessel_soft, _, orig_rgb = segmenter.segment(img_path)
    
    rgb_pil = Image.fromarray(orig_rgb)
    vessel_gray = (vessel_soft * 255).astype(np.uint8)
    vessel_pil = Image.fromarray(vessel_gray)
    
    rgb_t = rgb_transform(rgb_pil).unsqueeze(0)
    ves_t = vessel_transform(vessel_pil).unsqueeze(0)
    
    with torch.no_grad():
        efn_logit = efn_model(rgb_t, ves_t).item()
        dense_logit = dense_model(rgb_t, ves_t).item()
    
    efn_prob = torch.sigmoid(torch.tensor(efn_logit)).item()
    dense_prob = torch.sigmoid(torch.tensor(dense_logit)).item()
    
    return efn_logit, efn_prob, dense_logit, dense_prob

def main():
    if len(sys.argv) < 3:
        print("Kullanim: python diagnose3.py <hasta_img> <saglikli_img>")
        sys.exit(1)
    
    hasta_path = sys.argv[1]
    saglikli_path = sys.argv[2]
    
    print("Modeller yukleniyor...")
    segmenter = VesselSegmenter(unetpp_dir, device="cpu")
    efn = load_one_model(efn_dir, "efficientnet")
    dense = load_one_model(dense_dir, "densenet")
    
    # Define preprocessing variants to test
    preprocessing_variants = {}
    
    # Variant 1: Current (ImageNet norm, size 224)
    preprocessing_variants["1_ImageNet_224"] = (
        T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.5],[0.5])]),
    )
    
    # Variant 2: No normalization, just ToTensor (0-1), size 224
    preprocessing_variants["2_NoNorm_224"] = (
        T.Compose([T.Resize((224,224)), T.ToTensor()]),
        T.Compose([T.Resize((224,224)), T.ToTensor()]),
    )
    
    # Variant 3: ImageNet norm, size 512
    preprocessing_variants["3_ImageNet_512"] = (
        T.Compose([T.Resize((512,512)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        T.Compose([T.Resize((512,512)), T.ToTensor(), T.Normalize([0.5],[0.5])]),
    )
    
    # Variant 4: No norm, size 512
    preprocessing_variants["4_NoNorm_512"] = (
        T.Compose([T.Resize((512,512)), T.ToTensor()]),
        T.Compose([T.Resize((512,512)), T.ToTensor()]),
    )
    
    # Variant 5: ImageNet norm, size 256
    preprocessing_variants["5_ImageNet_256"] = (
        T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5],[0.5])]),
    )
    
    # Variant 6: Vessel also with ImageNet norm (in case trained that way with 3-ch repeat)
    preprocessing_variants["6_ImageNet_VesselSame_224"] = (
        T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485],[0.229])]),
    )
    
    # Variant 7: No normalization, size 384
    preprocessing_variants["7_NoNorm_384"] = (
        T.Compose([T.Resize((384,384)), T.ToTensor()]),
        T.Compose([T.Resize((384,384)), T.ToTensor()]),
    )
    
    # Variant 8: mean=0,std=1 for vessel (raw ToTensor is already 0-1)
    preprocessing_variants["8_ImageNet_VesselRaw_224"] = (
        T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        T.Compose([T.Resize((224,224)), T.ToTensor()]),
    )
    
    # Variant 9: ImageNet for both but size 512
    preprocessing_variants["9_ImageNet_VesselRaw_512"] = (
        T.Compose([T.Resize((512,512)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        T.Compose([T.Resize((512,512)), T.ToTensor()]),
    )
    
    # Variant 10: Size 224, ImageNet RGB, Vessel mean/std from typical vessel maps
    preprocessing_variants["10_ImageNet_VesselNorm02_224"] = (
        T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.2],[0.2])]),
    )
    
    print(f"\n{'='*90}")
    print(f"  PREPROCESSING KARSILASTIRMA TESTI")
    print(f"  Hasta dosya:    {os.path.basename(hasta_path)}")
    print(f"  Saglikli dosya: {os.path.basename(saglikli_path)}")
    print(f"{'='*90}")
    
    print(f"\n{'Variant':<35} | {'EFN-Hasta':>10} {'EFN-Sag':>10} {'FARK':>8} | {'DN-Hasta':>10} {'DN-Sag':>10} {'FARK':>8}")
    print("-" * 110)
    
    best_variant = None
    best_diff = 0
    
    for name, (rgb_tf, ves_tf) in preprocessing_variants.items():
        try:
            eh, ehp, dh, dhp = try_preprocessing(hasta_path, segmenter, efn, dense, name, rgb_tf, ves_tf)
            es, esp, ds, dsp = try_preprocessing(saglikli_path, segmenter, efn, dense, name, rgb_tf, ves_tf)
            
            efn_diff = abs(ehp - esp)
            dn_diff = abs(dhp - dsp)
            total_diff = efn_diff + dn_diff
            
            marker = " <-- EN IYI" if total_diff > best_diff else ""
            if total_diff > best_diff:
                best_diff = total_diff
                best_variant = name
            
            print(f"{name:<35} | {ehp:>10.4f} {esp:>10.4f} {efn_diff:>8.4f} | {dhp:>10.4f} {dsp:>10.4f} {dn_diff:>8.4f}{marker}")
        except Exception as e:
            print(f"{name:<35} | HATA: {str(e)[:60]}")
    
    print(f"\n{'='*90}")
    print(f"  EN AYIRT EDICI PREPROCESSING: {best_variant}  (toplam fark={best_diff:.4f})")
    print(f"{'='*90}")

if __name__ == "__main__":
    main()
