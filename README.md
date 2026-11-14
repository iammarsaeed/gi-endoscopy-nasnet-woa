# Gastrointestinal endoscopic image classification with fused NASNet-Mobile features and WOA feature selection

Code for the paper *Gastrointestinal Endoscopic Image Classification Using Fused NASNet-Mobile Features and Whale Optimization-Based Feature Selection*.

Pipeline, evaluated with five-fold cross-validation on the 8,000 images of Kvasir v2 (8 classes):

1. **Augmentation** of the training images: original, rotations by 90°, 180° and 270°, horizontal flip.
2. **BBHE** (brightness-preserving bi-histogram equalization) on the Y channel, giving an original and an enhanced stream.
3. **NASNet-Mobile**, fine-tuned inside each training fold; 1,056-D pooled features per stream.
4. **Serial fusion** of the two streams (2,112-D).
5. **Two-stage feature selection**: ReliefF keeps the 200 highest-ranked features; a binary Whale Optimization Algorithm (WOA) with an S-shaped transfer function (50 whales, 100 iterations) selects a subset of these.
6. **SVMs** (linear, quadratic, cubic, medium and coarse Gaussian) with test-time augmentation.

Folds are defined on the 8,000 **original** images. Augmentation, fine-tuning and feature selection use only the training images of each fold, so no information from a test image reaches training.

## Main results (mean ± SD over five folds)

| Configuration | Features | Accuracy (%) |
|---|---|---|
| Proposed: fused features, ReliefF + WOA, cubic SVM, test-time augmentation | 79 ± 14 | 91.8 ± 0.7 |
| Same, all 2,112 fused features | 2,112 | 92.5 ± 0.4 |
| Same feature selection on frozen ImageNet NASNet-Mobile features | 112 ± 28 | 80.5 ± 1.1 |
| ReliefF + random search with the same budget (5,050 evaluations) | 103 ± 5 | 91.5 ± 0.4 |
| ResNet-50 / EfficientNet-B0 / Swin-T, fine-tuned end to end on the same folds | – | 92.4 / 90.4 / 94.2 |

With the selected features, cubic-SVM training is about 18 times faster and prediction about 17 times faster than with all fused features.

## Data

Kvasir v2 is not included. Download it from the [official site](https://datasets.simula.no/kvasir/) and cite Pogorelov et al., *Kvasir: A multi-class image dataset for computer aided gastrointestinal disease detection*, ACM MMSys 2017. `prep.py` checks every image against the CRC32 index of the official archive, so any mirror can be used.

## Requirements

Python 3.11 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

TensorFlow (NASNet-Mobile) runs on the CPU. The PyTorch baselines use an Apple GPU (MPS) when available, otherwise CUDA or the CPU. The reported times were measured on an Apple M5 Pro (18-core CPU, 24 GB RAM).

## How to reproduce

Run all commands from the repository folder. Approximate times on the M5 Pro are in brackets.

**1. Data**

```bash
mkdir -p data results
# Index of the official archive (its last 2 MB), used to verify every image
curl -r 2487312085-2489312084 -o data/tail.bin https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip
# The archive itself (official server or any mirror)
python pget.py https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip 2489312085 data/kvasir.zip 24
python prep.py data/kvasir.zip data/tail.bin data/kvasir224.npz                                   # [~1 min]
```

**2. Folds** (shared by every experiment)

```bash
python -c "import json,numpy as np; from sklearn.model_selection import StratifiedKFold; \
y=np.load('data/kvasir224.npz')['y']; json.dump([[a.tolist(),b.tolist()] for a,b in \
StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y)],open('results/folds.json','w'))"
```

**3. Features and baselines**

```bash
python features.py data/kvasir224.npz data/feats.npy                                              # frozen features [~5 min]
python finetune_features.py data/kvasir224.npz results/folds.json 5                               # fine-tuned features [~80 min]
python baselines.py data/kvasir224.npz results/folds.json results/baselines.json 5 \
       resnet50 efficientnet_b0 swin_tiny_patch4_window7_224                                      # [~75 min, GPU]
```

**4. Experiments 1–3 and PCA** (no wrapper selection)

```bash
python run_cv_ft.py data/kvasir224.npz 'data/feats_ft_fold{k}.npy' results_ft 15                  # [~25 min]
cp results/baselines.json results_ft/
```

**5. Two-stage feature selection** (Experiment 4, ablation, comparison of selection methods, search budget)

```bash
python twostage_pipeline.py data/kvasir224.npz 'data/feats_ft_fold{k}.npy' data/feats.npy results_twostage 15   # [~30 min]
python search_budget.py data/kvasir224.npz 'data/feats_ft_fold{k}.npy' results_budget 12                         # [~30 min]
python report.py data/kvasir224.npz results_twostage results_twostage/figs
python twostage_stats.py data/kvasir224.npz results_twostage/stats.json
python fig_convergence.py results_budget results_twostage Fig12_search_convergence.png
```

**6. Figure 2(b)** (BBHE examples)

```bash
python fetch_samples.py data/tail.bin bbhe_samples
python bbhe_fig.py bbhe_samples Fig2b_BBHE_examples.png
```

## Files

| File | Purpose |
|---|---|
| `pget.py` | Parallel ranged download of the Kvasir v2 archive |
| `prep.py` | Verify the 8,000 images against the official CRC32 index; resize to 224 × 224; build the BBHE stream |
| `fetch_samples.py`, `bbhe_fig.py` | One image per class and the BBHE example figure |
| `features.py` | Frozen ImageNet NASNet-Mobile features (5 augmented versions × 2 streams) |
| `finetune_features.py` | Fine-tune NASNet-Mobile inside each training fold; extract 1,056-D pooled features |
| `run_cv.py` | Shared code: fitness function, BPSO, BGWO, GA, ReliefF, SVM presets, frozen-feature experiments |
| `run_cv_ft.py` | Experiments 1–3 and PCA on the fine-tuned features, with test-time augmentation |
| `bwoa.py` | Binary WOA with the S-shaped transfer function, and the random-search control |
| `twostage_pipeline.py` | All results that depend on feature selection, with the two-stage design |
| `search_budget.py` | WOA on all 2,112, the top 200 or the top 100 features, with 10 × 20 to 100 × 200 whales × iterations |
| `woa_budget.py` | Search budgets on all fused features with threshold binarization; provides the convergence-tracking fitness wrapper |
| `baselines.py` | ResNet-50, EfficientNet-B0 and Swin-T fine-tuned on the same folds |
| `report.py` | Tables, McNemar and paired tests, confusion matrices, training-time figures |
| `twostage_stats.py` | Further statistics used in the text |
| `fig_convergence.py` | Convergence of the wrapper searches (Figure 12) |

## Settings

- **Fine-tuning:** ImageNet NASNet-Mobile, dropout 0.2, 8-class softmax head; 5 epochs; Adam; batch size 32; cosine schedule with warm-up to 3 × 10⁻⁴; batches mix original and BBHE images 50/50 with a random rotation or flip.
- **ReliefF:** 1,000 sampled training images, 10 nearest neighbors per class, Manhattan distance on min–max-scaled features; top 200 kept.
- **Binary WOA:** binary positions; WOA update (encircling, random-whale exploration, spiral; b = 1) mapped to selection probabilities with T(v) = 1 / (1 + exp(−10(v − 0.5))); initial features selected with probability 0.5; 50 whales, 100 iterations.
- **Fitness:** 0.99 · error + 0.01 · |S|/D, where the error is that of a 5-NN classifier on an inner 80/20 split of the training originals of the fold. Seed 42 plus the fold index.
- **SVMs:** MATLAB Classification Learner presets; features standardized on the training fold; C = 1; one-vs-one.
- **Test-time augmentation:** SVM decision scores averaged over the five versions of each test image.

## Citation

The citation will be added when the paper is published.
