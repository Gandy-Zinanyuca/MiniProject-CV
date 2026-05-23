import numpy as np
from skimage import io, transform
from PIL import Image
import torch
from torchvision import transforms
from piq import brisque

#Relative path for the carpet where all the SR images are stored
base = "D:/2026-1/Computer Vision/mini/off/SinGAN/Output/SR/2/"

#Relative path for having access to the ground truth
original_path = "D:/2026-1/Computer Vision/mini/off/SinGAN/jcsmr.jpg"

# Generate bicubic baseline
original = io.imread(original_path)
H, W = original.shape[:2]
low_res = Image.fromarray(original).resize((W // 2, H // 2), Image.BICUBIC)
bicubic_up = np.array(low_res.resize((W, H), Image.BICUBIC))


def compute_brisque(img_array):
    img = img_array[:, :, :3].astype(np.float32) / 255.0
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        score = brisque(tensor, data_range=1.0)
    return score.item()


runs = {
    "Bicubic baseline": bicubic_up,
    "E2-A (noise=0.05)": io.imread(base + "build2_HR_amp0.05.png"),
    "E2-B (noise=0.10)": io.imread(base + "build2_HR_amp0.1.png"),
    "E2-C (noise=0.15)": io.imread(base + "build2_HR_amp0.15.png"),
    "E3-A (scale=0.80)": io.imread(base + "build2_HR_sr_f0.8.png"),
    "E3-B (scale=0.85)": io.imread(base + "build2_HR_amp0.1.png"),
    "E3-C (scale=0.90)": io.imread(base + "build2_HR_sr0.9.png"),
    "E4-B (nfc=64)": io.imread(base + "build2_HR_nfc64.png"),
}

print(f"{'Run':<25} {'PSNR':>8} {'SSIM':>8} {'BRISQUE':>10}")
print("-" * 55)

# Recompute PSNR and SSIM too so everything is in one table
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

gt = original[:, :, :3]

for name, img in runs.items():
    if img.shape[:2] != (H, W):
        img = (transform.resize(img, (H, W), anti_aliasing=True) * 255).astype(np.uint8)
    img3 = img[:, :, :3]

    p = psnr_fn(gt, img3, data_range=255)
    s = ssim_fn(gt, img3, channel_axis=2, data_range=255)
    b = compute_brisque(img3)
    print(f"{name:<25} {p:>8.2f} {s:>8.4f} {b:>10.4f}")