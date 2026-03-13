# =========================
# TASK2 (Hypertensive Retinopathy) - Vessel Soft Map Generation
# =========================

# 1) KURULUM: Subprocess yerine bunu kullanÄ±n.
# Versiyon hatalarÄ±nÄ± Ã¶nlemek iÃ§in kÄ±sÄ±tlamalarÄ± kaldÄ±rdÄ±k, en uyumlu sÃ¼rÃ¼mler yÃ¼klenecek.
!pip install segmentation-models-pytorch timm

import os, glob, random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from google.colab import drive

# ---- 0) Drive mount ----
if not os.path.exists("/content/drive"):
    drive.mount("/content/drive")
else:
    print("âœ… Drive already mounted")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("âœ… Device:", device)

# ---- 1) Paths (senin verdiÄŸin) ----
SEG_ROOT = "/content/drive/My Drive/Colab Notebooks/drive_retinal/TEZ/runs_segmentation"
CKPT_ROOT = os.path.join(SEG_ROOT, "unetpp_resnet34_crop512_seed42")

# GÃ–RÃœNTÃœ KLASÃ–RÃœNÃœ BURADAN KONTROL ET (Task 2 klasÃ¶rÃ¼n hangisiyse onu yaz)
IMG_DIR = "/content/drive/My Drive/Colab Notebooks/TEZ/hr_veri/images"
LABEL_CSV = "/content/drive/My Drive/Colab Notebooks/TEZ/hr_veri/hr_etiket.csv"

OUT_DIR = "/content/drive/My Drive/Colab Notebooks/TEZ/hr_veri/24aralikMaske"
os.makedirs(OUT_DIR, exist_ok=True)

ENCODER = "resnet34"
SIZE = 512
SAVE_PNG_PREVIEW = True
SHOW_SAMPLES = 3
SEED = 42
random.seed(SEED)

# ---- 2) Quick checks ----
if not os.path.exists(IMG_DIR):
    raise FileNotFoundError(f"âŒ IMG_DIR not found: {IMG_DIR}")
if not os.path.exists(LABEL_CSV):
    print(f"âš ï¸ LABEL CSV not found (you can still generate masks): {LABEL_CSV}")
else:
    df = pd.read_csv(LABEL_CSV)
    print("âœ… Label CSV loaded:", LABEL_CSV)
    print("   shape:", df.shape)

# Count images
exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
img_paths = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith(exts)])
print(f"ğŸ“· Task2 images found: {len(img_paths)}")
if len(img_paths) > 0:
    print("   example files:", [os.path.basename(p) for p in img_paths[:5]])
else:
    print("âš ï¸ HÄ°Ã‡ RESÄ°M BULUNAMADI! LÃ¼tfen IMG_DIR yolunu kontrol edin.")

# ---- 3) Model + Ensemble loader ----
def build_unetpp():
    return smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None
    )

def load_ensemble():
    models = []
    # 5 Fold'u yÃ¼kle
    for k in range(1, 6):
        ckpt_path = os.path.join(CKPT_ROOT, f"fold_{k}", "best.pth")

        if not os.path.exists(ckpt_path):
            print(f"âš ï¸ UYARI: Checkpoint bulunamadÄ±, atlanÄ±yor: {ckpt_path}")
            continue

        m = build_unetpp().to(device)
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            # EÄŸer checkpoint iÃ§inde 'model' anahtarÄ± varsa onu al, yoksa direkt state_dict'tir
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            m.load_state_dict(state_dict)
            m.eval()
            models.append(m)
        except Exception as e:
            print(f"âŒ Checkpoint yÃ¼klenirken hata: {ckpt_path}\n{e}")

    if len(models) == 0:
        raise RuntimeError("âŒ HiÃ§bir model yÃ¼klenemedi! LÃ¼tfen CKPT_ROOT yolunu kontrol edin.")

    print(f"âœ… Loaded {len(models)} checkpoints (ensemble).")
    return models

def read_green_clahe(img_path):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    g = rgb[..., 1]
    g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = g.astype(np.float32) / 255.0
    return rgb, g  # rgb(512,512,3), g(512,512)

@torch.no_grad()
def predict_ensemble(models, g_img):
    x = torch.from_numpy(g_img).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)
    probs = []
    for m in models:
        logits = m(x)
        prob = torch.sigmoid(logits)[0,0].detach().cpu().numpy()
        probs.append(prob)
    return np.mean(probs, axis=0).astype(np.float32)

def save_soft(out_dir, base, soft):
    npy_path = os.path.join(out_dir, base + "_vessel.npy")
    png_path = os.path.join(out_dir, base + "_vessel.png")
    np.save(npy_path, soft.astype(np.float32))
    if SAVE_PNG_PREVIEW:
        # GÃ¶rselleÅŸtirme iÃ§in 0-255 yap
        cv2.imwrite(png_path, (soft * 255).astype(np.uint8))
    return npy_path, png_path

# ---- 4) RUN generation ----
# Modelleri yÃ¼kle
models = load_ensemble()

if len(img_paths) > 0:
    print("\nğŸš€ Generating vessel maps for Task2...")
    for p in tqdm(img_paths):
        base = os.path.splitext(os.path.basename(p))[0]
        rgb, g = read_green_clahe(p)
        if g is None:
            print(f"âš ï¸ Okunamayan dosya: {p}")
            continue

        soft = predict_ensemble(models, g)
        save_soft(OUT_DIR, base, soft)

    print("âœ… Done. Saved to:", OUT_DIR)

    # ---- 5) Show sample comparisons (original vs soft vs overlay) ----
    print("\nğŸ–¼ï¸ Showing sample outputs...")
    sample_paths = random.sample(img_paths, k=min(SHOW_SAMPLES, len(img_paths)))

    for p in sample_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        rgb, g = read_green_clahe(p)

        # Soft map'i diskten okuyalÄ±m (kontrol amaÃ§lÄ±) veya memory'den alalÄ±m
        soft_path = os.path.join(OUT_DIR, base + "_vessel.png")
        if os.path.exists(soft_path):
            soft_png = cv2.imread(soft_path, cv2.IMREAD_GRAYSCALE)
            soft = soft_png.astype(np.float32) / 255.0
        else:
            continue

        # overlay on original RGB (red mask)
        overlay = rgb.copy()
        heat = (soft * 255).astype(np.uint8)
        # KÄ±rmÄ±zÄ± kanala ekle
        overlay[..., 0] = np.clip(overlay[..., 0] + heat, 0, 255)

        plt.figure(figsize=(18,5))
        plt.suptitle(f"Task2 | {os.path.basename(p)}", y=1.02)

        plt.subplot(1,3,1); plt.title("Original (RGB resized 512)"); plt.imshow(rgb); plt.axis("off")
        plt.subplot(1,3,2); plt.title("Vessel soft map"); plt.imshow(soft, cmap="gray"); plt.axis("off")
        plt.subplot(1,3,3); plt.title("Overlay (R+=softmap)"); plt.imshow(overlay); plt.axis("off")
        plt.show()

    print("âœ… Sample visualization complete.")
else:
    print("âŒ Ä°ÅŸlenecek gÃ¶rÃ¼ntÃ¼ bulunamadÄ±ÄŸÄ± iÃ§in iÅŸlem yapÄ±lmadÄ±.")

# --- CELL ---

import cv2
import matplotlib.pyplot as plt
import os
import random
import numpy as np

# Ayarlar (Ã–nceki koddan geldiÄŸi varsayÄ±lÄ±yor)
OUT_DIR = "/content/drive/My Drive/Colab Notebooks/TEZ/hr_veri/24aralikMaske"
IMG_DIR = "/content/drive/My Drive/Colab Notebooks/TEZ/hr_veri/images"
SIZE = 512
SHOW_SAMPLES = 5

# Resim yollarÄ±nÄ± tekrar alalÄ±m
exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
img_paths = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith(exts)])

print(f"âœ… Maskeler zaten Ã¼retilmiÅŸti. Åimdi {SHOW_SAMPLES} Ã¶rnek gÃ¶rselleÅŸtiriliyor...")

# Rastgele Ã¶rnekler seÃ§
sample_paths = random.sample(img_paths, k=min(SHOW_SAMPLES, len(img_paths)))

for p in sample_paths:
    base = os.path.splitext(os.path.basename(p))[0]

    # 1. Orijinal resmi oku ve RESIZE ET (HatayÄ± dÃ¼zelten kÄ±sÄ±m burasÄ±)
    bgr = cv2.imread(p)
    if bgr is None: continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) # 512x512 oldu

    # 2. ÃœretilmiÅŸ maskeyi diskten oku
    soft_path = os.path.join(OUT_DIR, base + "_vessel.png")
    if os.path.exists(soft_path):
        soft_png = cv2.imread(soft_path, cv2.IMREAD_GRAYSCALE)
        soft = soft_png.astype(np.float32) / 255.0
    else:
        print(f"Maske bulunamadÄ±: {base}")
        continue

    # 3. Overlay (ArtÄ±k boyutlar 512x512 olduÄŸu iÃ§in hata vermeyecek)
    overlay = rgb_resized.copy()
    heat = (soft * 255).astype(np.uint8)
    # KÄ±rmÄ±zÄ± kanala damarlarÄ± ekle (GÃ¶rsel efekt)
    overlay[..., 0] = np.clip(overlay[..., 0] + heat, 0, 255)

    # 4. Ã‡izdir
    plt.figure(figsize=(18,5))
    plt.suptitle(f"Task2 (Retinopati) | {os.path.basename(p)}", y=1.02)

    plt.subplot(1,3,1); plt.title("Original (512x512)"); plt.imshow(rgb_resized); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Vessel Soft Map"); plt.imshow(soft, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Overlay"); plt.imshow(overlay); plt.axis("off")
    plt.show()

print("âœ… GÃ¶rselleÅŸtirme tamamlandÄ±.")

# --- CELL ---

import os, glob

OUT_DIR = "/content/drive/My Drive/Colab Notebooks/TEZ/hr_veri/24aralikMaske"
IMG_DIR = "/content/drive/My Drive/Colab Notebooks/TEZ/hr_veri/images"

exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(exts)]
pngs = glob.glob(os.path.join(OUT_DIR, "*_vessel.png"))
npys = glob.glob(os.path.join(OUT_DIR, "*_vessel.npy"))

print("Task2 image count:", len(imgs))
print("Task2 vessel PNG count:", len(pngs))
print("Task2 vessel NPY count:", len(npys))

# hÄ±zlÄ± isim uyumu kontrolÃ¼ (kaÃ§ tanesi karÅŸÄ±lÄ±k geliyor)
img_bases = set(os.path.splitext(f)[0] for f in imgs)
png_bases = set(os.path.basename(f).replace("_vessel.png","") for f in pngs)
print("PNG matched to images:", len(img_bases & png_bases))


# --- CELL ---

# =========================
# TASK1 (Hypertension) - Vessel Soft Map Generation + Checks + Visualization
# UNet++ 5-fold ensemble (DRIVE-trained)
# =========================

# EÄŸer bir Ã¶nceki hÃ¼crede SMP kurulmadÄ±ysa aÃ§:
# !pip -q install segmentation-models-pytorch opencv-python tqdm pandas matplotlib

import os, random, glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from google.colab import drive

# ---- 0) Drive mount ----
if not os.path.exists("/content/drive"):
    drive.mount("/content/drive")
else:
    print("âœ… Drive already mounted")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("âœ… Device:", device)

# ---- 1) PATHS (senin verdiÄŸin) ----
SEG_ROOT = "/content/drive/My Drive/Colab Notebooks/drive_retinal/TEZ/runs_segmentation"
CKPT_ROOT = os.path.join(SEG_ROOT, "unetpp_resnet34_crop512_seed42")

IMG_DIR = "/content/drive/My Drive/Colab Notebooks/TEZ/hipertansif/images"
LABEL_CSV = "/content/drive/My Drive/Colab Notebooks/TEZ/hipertansif/csv/etiket.csv"

OUT_DIR = "/content/drive/My Drive/Colab Notebooks/TEZ/hipertansif/24aralikMaske"
os.makedirs(OUT_DIR, exist_ok=True)

ENCODER = "resnet34"
SIZE = 512
SAVE_PNG_PREVIEW = True
SHOW_SAMPLES = 5
SEED = 42
random.seed(SEED)

# ---- 2) Quick checks ----
if not os.path.exists(IMG_DIR):
    raise FileNotFoundError(f"âŒ IMG_DIR not found: {IMG_DIR}")

if os.path.exists(LABEL_CSV):
    df = pd.read_csv(LABEL_CSV)
    print("âœ… Label CSV loaded:", LABEL_CSV)
    print("   shape:", df.shape)
    print("   columns:", list(df.columns)[:10])
else:
    print("âš ï¸ Label CSV not found (mask generation will still run):", LABEL_CSV)

exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
imgs = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(exts)])
img_paths = [os.path.join(IMG_DIR, f) for f in imgs]

print(f"ğŸ“· Task1 images found: {len(img_paths)}")
print("   example files:", imgs[:5])

# ---- 3) Ensemble model loader ----
def build_unetpp():
    return smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None
    )

def load_ensemble():
    models = []
    for k in range(1, 6):
        ckpt_path = os.path.join(CKPT_ROOT, f"fold_{k}", "best.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"âŒ Missing checkpoint: {ckpt_path}")
        m = build_unetpp().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    print(f"âœ… Loaded {len(models)} checkpoints (ensemble).")
    return models

def read_green_clahe(img_path):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)

    g = rgb_resized[..., 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = g.astype(np.float32) / 255.0
    return rgb_resized, g

@torch.no_grad()
def predict_ensemble(models, g_img):
    x = torch.from_numpy(g_img).unsqueeze(0).unsqueeze(0).float().to(device)
    probs = []
    for m in models:
        logits = m(x)
        prob = torch.sigmoid(logits)[0,0].detach().cpu().numpy()
        probs.append(prob)
    return np.mean(probs, axis=0).astype(np.float32)

def save_soft(out_dir, base, soft):
    npy_path = os.path.join(out_dir, base + "_vessel.npy")
    png_path = os.path.join(out_dir, base + "_vessel.png")
    np.save(npy_path, soft.astype(np.float32))
    if SAVE_PNG_PREVIEW:
        cv2.imwrite(png_path, (soft * 255).astype(np.uint8))

# ---- 4) RUN generation ----
models = load_ensemble()

print("\nğŸš€ Generating vessel maps for Task1...")
for p in tqdm(img_paths):
    base = os.path.splitext(os.path.basename(p))[0]
    rgb, g = read_green_clahe(p)
    soft = predict_ensemble(models, g)
    save_soft(OUT_DIR, base, soft)

print("âœ… Done. Saved to:", OUT_DIR)

# ---- 5) Count outputs + name match ----
pngs = glob.glob(os.path.join(OUT_DIR, "*_vessel.png"))
npys = glob.glob(os.path.join(OUT_DIR, "*_vessel.npy"))

img_bases = set(os.path.splitext(f)[0] for f in imgs)
png_bases = set(os.path.basename(f).replace("_vessel.png","") for f in pngs)

print("\nğŸ“Š OUTPUT COUNTS")
print("Task1 image count:", len(imgs))
print("Task1 vessel PNG count:", len(pngs))
print("Task1 vessel NPY count:", len(npys))
print("PNG matched to images:", len(img_bases & png_bases))

# ---- 6) Show sample outputs ----
print("\nğŸ–¼ï¸ Showing sample outputs...")
sample_paths = random.sample(img_paths, k=min(SHOW_SAMPLES, len(img_paths)))

for p in sample_paths:
    base = os.path.splitext(os.path.basename(p))[0]
    rgb, g = read_green_clahe(p)

    soft_path = os.path.join(OUT_DIR, base + "_vessel.png")
    soft_png = cv2.imread(soft_path, cv2.IMREAD_GRAYSCALE)
    soft = soft_png.astype(np.float32) / 255.0

    overlay = rgb.copy()
    heat = (soft * 255).astype(np.uint8)
    overlay[..., 0] = np.clip(overlay[..., 0] + heat, 0, 255)

    plt.figure(figsize=(18,5))
    plt.suptitle(f"Task1 (Hypertension) | {os.path.basename(p)}", y=1.02)
    plt.subplot(1,3,1); plt.title("Original (512x512)"); plt.imshow(rgb); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Vessel Soft Map"); plt.imshow(soft, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Overlay"); plt.imshow(overlay); plt.axis("off")
    plt.show()

print("âœ… Task1 visualization complete.")


# --- CELL ---

import os, glob

IMG_DIR = "/content/drive/My Drive/Colab Notebooks/TEZ/hipertansif/images"
OUT_DIR = "/content/drive/My Drive/Colab Notebooks/TEZ/hipertansif/24aralikMaske"

exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(exts)]
pngs = glob.glob(os.path.join(OUT_DIR, "*_vessel.png"))
npys = glob.glob(os.path.join(OUT_DIR, "*_vessel.npy"))

img_bases = set(os.path.splitext(f)[0] for f in imgs)
png_bases = set(os.path.basename(f).replace("_vessel.png","") for f in pngs)

print("Task1 image count:", len(imgs))
print("Task1 vessel PNG count:", len(pngs))
print("Task1 vessel NPY count:", len(npys))
print("PNG matched to images:", len(img_bases & png_bases))


# --- CELL ---

