# Single-Image Super-Resolution with SinGAN: An Empirical Study

A Computer Vision mini-project investigating single-image super-resolution using SinGAN, with systematic ablation studies of noise, scale, and capacity parameters.

## Project Overview

This project addresses the challenge of recovering high-resolution images from a single low-resolution photograph without requiring large-scale datasets. We apply **SinGAN**, a multi-scale generative adversarial network, to empirically study how three key parameters affect super-resolution quality:

- **noise_amp**: Controls noise magnitude and sharpness
- **scale_factor**: Determines pyramid depth
- **nfc**: Sets convolutional channel capacity

### Key Findings

- **E4-B (nfc=64)** achieves the best performance with **PSNR=34.09 dB** and **SSIM=0.9729**, surpassing bicubic baseline by 5.43 dB
- Channel capacity is the dominant factor in SR quality
- Sufficient noise is required to synthesize meaningful high-frequency detail
- SinGAN produces more perceptually natural outputs than interpolation (confirmed by BRISQUE scores)
- Distortion and perceptual metrics reveal complementary but sometimes contradictory aspects of image quality

## Project Structure

```
MiniProject-CV/
├── src/
│   └── Metrics.py          # Evaluation script computing PSNR, SSIM, BRISQUE metrics
├── data/                   # Input images and datasets
├── report/
│   └── CV2026_Mini_Project_Template (3).pdf  # Full project report
├── requirements.txt        # Python dependencies
├── LICENSE                 # Project license
└── README.md              # This file
```

## Method

### SinGAN Architecture

SinGAN learns from a single image by training a multi-scale pyramid of patch-GANs. For each scale *n*:

- **Generator (Gn)**: Upsamples the previous scale's output and adds learned noise
- **Discriminator (Dn)**: Distinguishes real patches from generated patches

The generator minimizes:
```
L(Gn, Dn) = Ladv(Gn, Dn) + α·Lrec(Gn)
```

Where Ladv is WGAN-GP adversarial loss and Lrec is reconstruction loss.

### Experimental Setup

- **Input image**: JCSMR building photograph (5616×3744 px, downsampled to 256×256)
- **Upscaling factor**: sr_factor = 2
- **Training iterations**: 500 per scale (enables rapid parameter comparison)
- **Baseline**: Bicubic interpolation (PSNR=28.66 dB, SSIM=0.8932, BRISQUE=38.08)

### Ablation Study

Three controlled experiments, each varying one parameter:

| Parameter | Values | Baseline |
|-----------|--------|----------|
| noise_amp | {0.05, 0.10, 0.15} | 0.10 |
| scale_factor | {0.80, 0.85, 0.90} | 0.85 |
| nfc | {32, 64} | 32 |

## Evaluation Metrics

Three complementary metrics assess image quality:

1. **PSNR** (Peak Signal-to-Noise Ratio): Pixel-level distortion metric
2. **SSIM** (Structural Similarity Index): Structural similarity metric
3. **BRISQUE** (Blind/Referenceless Image Spatial Quality Evaluator): No-reference perceptual quality metric

Results show metric disagreement—for example, E3-C achieves high SSIM despite visible color drift, while BRISQUE correctly penalizes it.

## Results Summary

| Configuration | PSNR (dB) | SSIM | BRISQUE | Notes |
|---------------|-----------|------|---------|-------|
| Bicubic baseline | 28.66 | 0.8932 | 38.08 | Reference |
| E2-A (noise=0.05) | 27.08 | 0.8925 | 18.13 | Below baseline PSNR |
| E2-B (noise=0.10) | 27.13 | 0.9349 | 17.78 | Baseline noise |
| E2-C (noise=0.15) | 29.47 | 0.9356 | 16.63 | Exceeds baseline PSNR |
| E3-A (scale=0.80) | 30.54 | 0.9629 | 19.36 | Best scale factor |
| E3-B (scale=0.85) | 27.13 | 0.9349 | 17.78 | Baseline scale |
| E3-C (scale=0.90) | 28.89 | 0.9553 | 22.91 | Color drift observed |
| E4-A (nfc=32) | 27.13 | 0.9349 | 17.78 | Baseline capacity |
| E4-B (nfc=64) | **34.09** | **0.9729** | 18.27 | **Best performance** |

## Installation

### Requirements

- Python 3.8+
- PyTorch
- scikit-image
- PIL/Pillow
- piq (for BRISQUE)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
python src/Metrics.py
```

## Usage

The `Metrics.py` script evaluates all configured SinGAN variants against the ground truth:

```bash
python src/Metrics.py
```

Output includes a table with PSNR, SSIM, and BRISQUE scores for each configuration.

## Key Insights

### Parameter Effects

1. **Noise Amplitude**: Shows a threshold effect—only noise_amp=0.15 exceeds bicubic baseline PSNR, suggesting minimum noise required for high-frequency synthesis
2. **Scale Factor**: Non-monotonic results with potential color artifacts at higher values (scale=0.90)
3. **Channel Capacity**: Dominant factor—nfc=64 dramatically improves quality over nfc=32

### Metric Disagreement

A critical finding is disagreement between metrics:
- SSIM insensitive to color shifts (inconsistent with perceptual quality)
- BRISQUE correctly penalizes perceptual artifacts
- All SinGAN configurations achieve lower BRISQUE than bicubic, demonstrating more natural outputs despite potential pixel-level inaccuracy

## Societal Implications & Limitations

### Applications
- Satellite imaging
- Historical image restoration
- Medical diagnostics

### Ethical Concerns
- **Hallucination risk**: Generative methods invent rather than recover detail, problematic in forensic contexts
- **Privacy**: Enhanced SR can identify previously anonymous individuals
- **Architectural bias**: Model may fail on faces or unfamiliar scene types

### Technical Limitations
- 500 iterations per scale may underfit compared to default 2000
- Single-image training limits generalization
- Evaluation on single image; results may not transfer to other scenes

## Future Work

- Incorporate perceptual loss functions
- Improve generalization across diverse image types
- Increase training iterations for quality ceiling estimation
- Expand to multiple images for robustness analysis
- Address ethical concerns in forensic applications

## References

[1] Distortion vs. perception tradeoff in image quality assessment  
[2] SinGAN: Learning a Generative Model from a Single Natural Image  
[3] Additional references in project report

## Author

**Gandy Zinanyuca**  
COMP/ENGN 4528/6528 Computer Vision  
Student ID: u8430326

## License

See LICENSE file for details.

---

For detailed methodology, experimental results, and discussion, refer to the full report in `report/CV2026_Mini_Project_Template (3).pdf`.