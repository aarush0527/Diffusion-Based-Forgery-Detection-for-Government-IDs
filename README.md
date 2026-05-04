# Diffusion-Based Forgery Detection on Identity Documents

A research pipeline that uses Denoising Diffusion Probabilistic Models (DDPM) for anomaly detection on the FantasyID dataset, followed by an EfficientNet-B0 classifier to distinguish genuine ID cards from digitally forged ones.

---

## Overview

Most forgery detection systems try to learn what fake looks like. This project takes the opposite approach — it learns what real looks like, and flags anything that deviates from it.

A DDPM is trained exclusively on genuine (bonafide) ID card images. When a forged document is passed through this model for reconstruction, the tampered regions produce large pixel-level errors because the model has never learned to reconstruct forgeries. These errors are captured as residual maps (anomaly heatmaps), which are then fed into a fine-tuned EfficientNet-B0 classifier for the final real/fake decision.

This two-stage design makes the classifier language-agnostic and device-agnostic — it only sees the anomaly signal, not the raw visual diversity of 13 language templates and 3 capture devices.

---

## Dataset

**FantasyID** — Korshunov et al., IJCB 2025

| Split | Bonafide | Attack | Total |
|-------|----------|--------|-------|
| Train | 633 | 1,266 | 1,899 |
| Test  | 300 | 1,085 | 1,385 |

**Attack types:**
- `digital_1 / digital_2 / digital_3` — digital splicing (region replacement)
- `facedancer` — deep learning-based face swap
- `textdiffuserft_bfei` — text inpainting via diffusion

**Languages:** Arabic, Chinese, French, Hindi, Portuguese, Russian, Turkish, Ukrainian, Singaporean, Dutch, Hong Kong, English, Persian

**Capture devices (bonafide):** Flatbed scanner, Huawei smartphone, iPhone 15 Pro

Dataset available at: https://zenodo.org/records/17063366

---

## Pipeline

```
FantasyID Images
    │
    ├── Bonafide only ──► PHASE 3: DDPM Training
    │                     (learns genuine document distribution)
    │
    └── All images ──────► PHASE 4: Partial Diffusion Reconstruction
                           (add 50% noise → denoise → |original - reconstruction|)
                           → Residual Maps
                               │
                               └──► PHASE 5: EfficientNet-B0 Classifier
                                    → Real / Fake
```

### Why partial diffusion (t_mid = 500)?

Adding full noise destroys all image structure. Adding no noise changes nothing. At 50% noise, the DDPM retains enough document structure to reconstruct genuine images faithfully, while tampered regions in forged images get pulled toward the genuine manifold — creating visible reconstruction artifacts exactly where the forgery is.

---

## Model Architecture

**DDPM — UNet2DModel (HuggingFace Diffusers)**
- 4 resolution levels: 128 → 64 → 32 → 16
- Self-attention at 32×32 and 16×16 (captures global document layout)
- 2 ResNet blocks per level
- ~25M parameters
- DDPMScheduler: 1000 timesteps, linear beta schedule (0.0001 → 0.02)
- Trained for 200 epochs on bonafide images only

**Classifier — EfficientNet-B0 (timm)**
- ImageNet-pretrained backbone (feature extractor frozen initially)
- Custom head: `1280 → BatchNorm → Dropout → Linear(256) → ReLU → Linear(2)`
- Input: residual maps (not raw images)
- Trained for 40 epochs with OneCycleLR scheduler

---

## Handling Class Imbalance

The training set has a 2:1 attack-to-genuine ratio. Two strategies are applied simultaneously:

- **WeightedRandomSampler** — genuine images are oversampled during training so each batch is approximately balanced
- **Class-weighted CrossEntropy** — the loss function penalizes mistakes on the minority class (bonafide) more heavily

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Of predicted attacks, how many were actually attacks |
| Recall | Of actual attacks, how many were caught |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Threshold-independent discrimination ability |
| Per-attack accuracy | Breakdown by each forgery type |

Recall is the most critical metric in a security context — missing a forgery is more costly than a false alarm.

---

## Visualizations Produced

- Dataset EDA: class distribution, attack subtypes, language breakdown, capture devices
- DDPM training loss curve (raw + smoothed)
- Reconstruction comparison: original → reconstructed → residual heatmap → anomaly score
- Residual score distribution: genuine vs. forged (histogram + boxplot + per-subtype)
- Classifier training curves: loss, accuracy, AUC over epochs
- Confusion matrix (normalized) and ROC curve with optimal threshold
- Per-attack-type accuracy bar chart
- Qualitative anomaly analysis grids for bonafide and attack samples

---

## Setup and Usage

**Requirements**
```bash
pip install diffusers accelerate transformers huggingface_hub timm scikit-learn seaborn tqdm scipy torch torchvision
```

**Run on Google Colab (recommended)**

The notebook is designed for Colab with a T4 or A100 GPU.

1. Upload `UG_com.ipynb` to Google Colab
2. Mount Google Drive and place `FantasyID.tgz` in your Drive root, or let the notebook download it directly from Zenodo
3. Run all cells in order (Phases 1 through 6)

**Alternatively, run on Kaggle**

The notebook also supports Kaggle with dataset paths pre-configured.

---

## Project Structure

```
outputs/
├── checkpoints/
│   ├── ddpm_best.pt          # Best DDPM checkpoint (lowest training loss)
│   ├── ddpm_epoch50.pt       # Periodic checkpoints
│   └── classifier_best.pt    # Best classifier checkpoint (highest AUC)
├── residuals/                # Residual maps mirroring FantasyID structure
│   ├── train/bonafide/
│   ├── train/attack/
│   ├── test/bonafide/
│   └── test/attack/
├── visualizations/           # All plots and figures
└── manifest.csv              # Full dataset manifest with labels and metadata
```

---

## Key Design Decisions and Tradeoffs

**Why not train a classifier directly on raw images?**
Raw images vary enormously across 13 languages and 3 capture devices. A model trained this way risks learning language-specific or device-specific patterns rather than forgery-specific ones. Residual maps collapse this variability — both a genuine Chinese scanned ID and a genuine Arabic iPhone photo should produce similarly low, uniform residuals.

**Why EfficientNet-B0 and not a larger model?**
With only ~1,900 training images total and residual maps as input (which are far simpler than natural images), a lightweight pretrained model is sufficient and significantly reduces overfitting risk.

**Why DDPM over VAE or GAN for reconstruction?**
VAEs tend to produce blurry reconstructions, which would obscure the residual signal. GANs are unstable to train and prone to mode collapse. DDPMs produce high-fidelity reconstructions with stable training, which is precisely what anomaly detection via residuals requires.

---

## Limitations

- The DDPM is trained on only 633 genuine images, which is small for a generative model. Performance may degrade on language/device combinations that are underrepresented in training data.
- The t_mid = 500 hyperparameter is not tuned via ablation. Different values may produce better separation between genuine and forged residuals.
- The comparison baseline in the notebook is configured with intentionally suboptimal settings to illustrate worst-case naive performance, not as a rigorous ablation.

---

## References

- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
- Korshunov et al., "FantasyID", IJCB 2025
- Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019
- HuggingFace Diffusers: https://github.com/huggingface/diffusers

---

## Author

Built as an undergraduate research project exploring generative models for document forensics.
