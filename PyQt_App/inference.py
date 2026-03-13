"""
Hypertensive Retinopathy Detection — Inference Engine
=====================================================
1. UNet++ (smp) vessel segmentation  — 5-fold ensemble
2. Dual-backbone classification      — 5-fold ensemble
   • rgb_backbone   → 3-ch fundus image
   • vessel_backbone → 1-ch vessel soft-map
   • head           → binary (hypertensive / healthy)
"""

import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import timm
import segmentation_models_pytorch as smp
from PIL import Image
import cv2
import numpy as np


# ═══════════════════ Vessel Segmentation (UNet++) ═══════════════════

class VesselSegmenter:
    """5-fold UNet++ ensemble for retinal vessel segmentation."""

    def __init__(self, ckpt_dir, device="cpu", crop_size=512):
        self.device = device
        self.crop_size = crop_size
        self.models = []

        for k in range(1, 6):
            ckpt_path = os.path.join(ckpt_dir, f"fold_{k}", "best.pth")
            if not os.path.exists(ckpt_path):
                print(f"[WARN] Segmentasyon checkpoint bulunamadı: {ckpt_path}")
                continue

            model = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=1,
                classes=1,
                activation=None,
            )
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            self.models.append(model)

        if not self.models:
            raise RuntimeError(f"UNet++ checkpoint'ları yüklenemedi: {ckpt_dir}")
        print(f"[OK] {len(self.models)} UNet++ segmentasyon modeli yüklendi.")

    @torch.no_grad()
    def segment(self, image_path):
        """
        Returns:
            vessel_soft  : (H, W) float32 vessel probability map [0,1] at crop_size
            vessel_vis   : (H, W, 3) uint8 visualization (green overlay)
            orig_rgb     : (H, W, 3) uint8 original image resized to crop_size
        """
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Resim okunamadı: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (self.crop_size, self.crop_size))

        # Green channel + CLAHE
        g = rgb_resized[..., 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_clahe = clahe.apply(g).astype(np.float32) / 255.0

        # Predict with ensemble
        x = torch.from_numpy(g_clahe).unsqueeze(0).unsqueeze(0).float().to(self.device)
        probs = []
        for m in self.models:
            logits = m(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            probs.append(prob)
        vessel_soft = np.mean(probs, axis=0).astype(np.float32)

        # Create visualization: green overlay on original
        vessel_uint8 = (vessel_soft * 255).astype(np.uint8)
        vessel_vis = rgb_resized.copy()
        vessel_vis[..., 1] = np.clip(
            vessel_vis[..., 1].astype(np.int16) + (vessel_uint8 * 0.6).astype(np.int16), 0, 255
        ).astype(np.uint8)

        return vessel_soft, vessel_vis, rgb_resized


# ═══════════════════ Classification (Dual-Backbone) ═══════════════════

class DualDenseNet121(nn.Module):
    """Two DenseNet-121 backbones (RGB 3ch + Vessel 1ch) → concat → Linear head."""

    def __init__(self):
        super().__init__()
        self.rgb_backbone = tv_models.densenet121(weights=None)
        self.vessel_backbone = tv_models.densenet121(weights=None)
        self.vessel_backbone.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_ftrs = self.rgb_backbone.classifier.in_features  # 1024
        self.rgb_backbone.classifier = nn.Identity()
        self.vessel_backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs * 2, 1),
        )

    def forward(self, rgb, vessel):
        f_rgb = self.rgb_backbone(rgb)
        f_ves = self.vessel_backbone(vessel)
        return self.head(torch.cat([f_rgb, f_ves], dim=1))


class DualEfficientNetB0(nn.Module):
    """Two EfficientNet-B0 backbones (timm) (RGB 3ch + Vessel 1ch) → concat → Linear head."""

    def __init__(self):
        super().__init__()
        self.rgb_backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0, in_chans=3
        )
        self.vessel_backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0, in_chans=1
        )
        num_ftrs = self.rgb_backbone.num_features  # 1280
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs * 2, 1),
        )

    def forward(self, rgb, vessel):
        f_rgb = self.rgb_backbone(rgb)
        f_ves = self.vessel_backbone(vessel)
        return self.head(torch.cat([f_rgb, f_ves], dim=1))


# ═══════════════════ Main Predictor ═══════════════════

class EnsemblePredictor:
    """Full pipeline: vessel segmentation → dual-backbone classification."""

    def __init__(self, efn_dir, dense_dir, unetpp_dir, device="cpu"):
        self.device = device

        # 1. Vessel segmenter
        self.segmenter = VesselSegmenter(unetpp_dir, device=device)

        # 2. Classification models (5-fold each)
        self.efn_models = self._load_cls_folds(efn_dir, "efficientnet", device)
        self.dense_models = self._load_cls_folds(dense_dir, "densenet", device)

        # Transforms for classification
        # NOT: Egitimde normalizasyon KULLANILMAMIS, sadece ToTensor (0-255 -> 0-1)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.vessel_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    @staticmethod
    def _load_cls_folds(folder, arch_type, device):
        models = []
        for k in range(1, 6):
            path = os.path.join(folder, f"fold{k}_best.pth")
            if not os.path.exists(path):
                print(f"[WARN] Checkpoint bulunamadı: {path}")
                continue
            model = DualEfficientNetB0() if arch_type == "efficientnet" else DualDenseNet121()
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            models.append(model)
        if not models:
            raise RuntimeError(f"Hiçbir checkpoint yüklenemedi: {folder}")
        print(f"[OK] {len(models)} {arch_type} sınıflandırma modeli yüklendi.")
        return models

    @torch.no_grad()
    def predict(self, image_path, model_choice="Ensemble"):
        """
        Full pipeline:
        1. Segment vessels (UNet++)
        2. Classify with dual-backbone

        Returns: (probability, orig_rgb_512, vessel_vis_512, vessel_soft_512)
        """
        # Step 1: Vessel segmentation
        vessel_soft, vessel_vis, orig_rgb = self.segmenter.segment(image_path)

        # Step 2: Prepare classification inputs
        rgb_pil = Image.fromarray(orig_rgb)
        vessel_gray = (vessel_soft * 255).astype(np.uint8)
        vessel_pil = Image.fromarray(vessel_gray)

        rgb_t = self.rgb_transform(rgb_pil).unsqueeze(0).to(self.device)
        ves_t = self.vessel_transform(vessel_pil).unsqueeze(0).to(self.device)

        # Step 3: Classification
        def _ensemble_prob(model_list):
            probs = []
            for m in model_list:
                logit = m(rgb_t, ves_t)
                probs.append(torch.sigmoid(logit).item())
            return float(np.mean(probs))

        if model_choice == "Ensemble":
            p_efn = _ensemble_prob(self.efn_models) if self.efn_models else 0.0
            p_dense = _ensemble_prob(self.dense_models) if self.dense_models else 0.0
            n = (1 if self.efn_models else 0) + (1 if self.dense_models else 0)
            final = (p_efn + p_dense) / max(n, 1)
        elif model_choice == "EfficientNet":
            final = _ensemble_prob(self.efn_models)
        else:
            final = _ensemble_prob(self.dense_models)

        # Sigmoid ciktisi P(Hasta) veriyor (label=1 = hasta).
        return final, orig_rgb, vessel_vis, vessel_soft
