# Gastrointestinal endoscopic image classification with fused NASNet-Mobile features and WOA feature selection

Code for the paper *Gastrointestinal Endoscopic Image Classification Using Fused NASNet-Mobile Features and Whale Optimization-Based Feature Selection*. It contains everything needed to reproduce the paper's tables and figures from the raw dataset.

## Dataset

The experiments use **Kvasir v2**, a public dataset of conventional gastrointestinal endoscopy images (gastroscopy and colonoscopy) collected at Vestre Viken Health Trust, Norway:

- 8,000 images, 8 balanced classes of 1,000 images each;
- resolutions from 720 × 576 to 1920 × 1072 pixels (resized to 224 × 224 here);
- classes: anatomical landmarks (normal Z-line, normal pylorus, normal cecum), pathological findings (esophagitis, polyps, ulcerative colitis) and polyp-removal images (dyed and lifted polyps, dyed resection margins);
- no patient identifiers, so the folds are defined at image level.

The dataset is **not included** in this repository. Download it from the official page, https://datasets.simula.no/kvasir/ (archive `kvasir-dataset-v2.zip`, 2,489,312,085 bytes). Kvasir may be used for research and educational purposes; if you use it, cite:

> K. Pogorelov et al., "Kvasir: A multi-class image dataset for computer aided gastrointestinal disease detection," in *Proc. 8th ACM Multimedia Systems Conference (MMSys)*, 2017, pp. 164–169.

`data_prep/prep.py` checks every image against the CRC32 index of the official archive, so a copy from a mirror is accepted only if it is identical to the original.

## Method

1. **Augmentation** of the training images: original, rotations by 90°, 180° and 270°, horizontal flip.
2. **BBHE** (brightness-preserving bi-histogram equalization) on the Y channel, giving an original and an enhanced stream.
3. **NASNet-Mobile**, fine-tuned inside each training fold; 1,056-D pooled features per stream.
4. **Serial fusion** of the two streams (2,112-D).
5. **Two-stage feature selection**: ReliefF keeps the 200 highest-ranked features; a binary Whale Optimization Algorithm (WOA) with an S-shaped transfer function (50 whales, 100 iterations) selects a subset of these.
6. **SVMs** (linear, quadratic, cubic, medium and coarse Gaussian) with test-time augmentation.

All experiments use five-fold cross-validation defined on the 8,000 **original** images. Augmentation, fine-tuning and feature selection use only the training images of each fold, so no information from a test image reaches training.

## Main results (mean ± SD over five folds)

| Configuration | Features | Accuracy (%) |
|---|---|---|
| Proposed: fused features, ReliefF + WOA, cubic SVM, test-time augmentation | 79 ± 14 | 91.8 ± 0.7 |
| Same, all 2,112 fused features | 2,112 | 92.5 ± 0.4 |
| Same feature selection on frozen ImageNet NASNet-Mobile features | 112 ± 28 | 80.5 ± 1.1 |
| ReliefF + random search with the same budget (5,050 evaluations) | 103 ± 5 | 91.5 ± 0.4 |
| ResNet-50 / EfficientNet-B0 / Swin-T, fine-tuned end to end on the same folds | – | 92.4 / 90.4 / 94.2 |

With the selected features, cubic-SVM training is about 18 times faster and prediction about 17 times faster than with all fused features.

## Repository structure

```
data_prep/
  pget.py                  parallel ranged download of the Kvasir v2 archive
  prep.py                  CRC32 check against the official archive, resize to 224 x 224, BBHE stream
  fetch_samples.py         one image per class from the official archive (for Fig. 2b)
feature_extraction/
  features.py              frozen ImageNet NASNet-Mobile features (5 augmented versions x 2 streams)
  finetune_features.py     NASNet-Mobile fine-tuned inside each training fold; 1,056-D pooled features
cross_validation/
  run_cv.py                shared code: fitness function, binary WOA (threshold), BPSO, BGWO, GA, ReliefF, SVM presets;
                           frozen-feature experiments
  run_cv_ft.py             Experiments 1-3 and PCA on the fine-tuned features, with test-time augmentation
  leaky_check.py           control experiment: augmentation before the split, to measure data leakage
feature_selection/
  bwoa.py                  binary WOA with the S-shaped transfer function, and the random-search control
  twostage_pipeline.py     all results that depend on feature selection, with the two-stage ReliefF -> WOA design
  search_budget.py         WOA on all 2,112, the top 200 or the top 100 features, 10 x 20 to 100 x 200 whales x iterations
  woa_budget.py            search budgets on all fused features with threshold binarization; convergence-tracking fitness
deep_baselines/
  baselines.py             ResNet-50, EfficientNet-B0 and Swin-T fine-tuned on the same folds
analysis/
  report.py                tables, McNemar and paired tests, confusion matrices, training-time figures (Figs. 4-11)
  twostage_stats.py        further statistics used in the text
figures/
  fig1_pipeline.py         Fig. 1, pipeline and evaluation protocol
  bbhe_fig.py              Fig. 2(b), BBHE examples with luminance histograms
  fig_convergence.py       Fig. 12, convergence of the wrapper searches
scripts/
  reproduce.sh             runs the whole pipeline in order
```

## Requirements

Python 3.11 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

TensorFlow (NASNet-Mobile) runs on the CPU. The PyTorch baselines use an Apple GPU (MPS) when available, otherwise CUDA or the CPU. The reported times were measured on an Apple M5 Pro (18-core CPU, 24 GB RAM).

## How to reproduce

Run everything from the repository root; scripts are run as modules (`python -m folder.script`). The whole pipeline is in one script:

```bash
bash scripts/reproduce.sh
```

It takes about six hours on the M5 Pro and writes the tables to `results_twostage/summary.md`, the statistics to `results_twostage/stats.json` and the figures to `results_twostage/figs/`. The individual steps are:

| Step | Command | Time |
|---|---|---|
| 1. Data | `python -m data_prep.pget <url> 2489312085 data/kvasir.zip 24`, then `python -m data_prep.prep data/kvasir.zip data/tail.bin data/kvasir224.npz` | download + 1 min |
| 2. Folds | stratified 5-fold split with seed 42, saved to `results/folds.json` (see `scripts/reproduce.sh`) | seconds |
| 3. Features | `python -m feature_extraction.features ...` and `python -m feature_extraction.finetune_features data/kvasir224.npz results/folds.json 5` | 5 + 80 min |
| 4. Baselines | `python -m deep_baselines.baselines data/kvasir224.npz results/folds.json results/baselines.json 5 resnet50 efficientnet_b0 swin_tiny_patch4_window7_224` | 75 min (GPU) |
| 5. Experiments 1–3 | `python -m cross_validation.run_cv_ft data/kvasir224.npz 'data/feats_ft_fold{k}.npy' results_ft 15` | 25 min |
| 6. Two-stage selection | `python -m feature_selection.twostage_pipeline ... results_twostage 15` and `python -m feature_selection.search_budget ... results_budget 12` | 30 + 30 min |
| 7. Tables and figures | `python -m analysis.report`, `python -m analysis.twostage_stats`, `python -m figures.*` | minutes |

The control experiment `cross_validation/leaky_check.py` is not part of `reproduce.sh`; run it with `python -m cross_validation.leaky_check data/kvasir224.npz data/feats.npy results/leaky_control.json 5` after step 3.

## Settings

- **Fine-tuning:** ImageNet NASNet-Mobile, dropout 0.2, 8-class softmax head; 5 epochs; Adam; batch size 32; cosine schedule with warm-up to 3 × 10⁻⁴; batches mix original and BBHE images 50/50 with a random rotation or flip.
- **ReliefF:** 1,000 sampled training images, 10 nearest neighbors per class, Manhattan distance on min–max-scaled features; top 200 kept.
- **Binary WOA:** binary positions; WOA update (encircling, random-whale exploration, spiral; b = 1) mapped to selection probabilities with T(v) = 1 / (1 + exp(−10(v − 0.5))); initial features selected with probability 0.5; 50 whales, 100 iterations.
- **Fitness:** 0.99 · error + 0.01 · |S|/D, where the error is that of a 5-NN classifier on an inner 80/20 split of the training originals of the fold. Seed 42 plus the fold index.
- **SVMs:** MATLAB Classification Learner presets; features standardized on the training fold; C = 1; one-vs-one.
- **Test-time augmentation:** SVM decision scores averaged over the five versions of each test image.

## Citation

The citation will be added when the paper is published.
