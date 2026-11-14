#!/usr/bin/env bash
# Full reproduction of the paper's results, from the raw Kvasir v2 archive to the tables and figures.
# Run from the repository root:  bash scripts/reproduce.sh
# Approximate times on an Apple M5 Pro (18-core CPU) are given in brackets; the whole run takes about 6 hours.
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-15}
URL=https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip
mkdir -p data results

# 1. Data: download, verify every image against the official CRC32 index, resize, build the BBHE stream  [download + ~1 min]
[ -f data/tail.bin ] || curl -r 2487312085-2489312084 -o data/tail.bin "$URL"
[ -f data/kvasir.zip ] || python -m data_prep.pget "$URL" 2489312085 data/kvasir.zip 24
[ -f data/kvasir224.npz ] || python -m data_prep.prep data/kvasir.zip data/tail.bin data/kvasir224.npz

# 2. Five stratified folds on the 8,000 original images (shared by every experiment)
[ -f results/folds.json ] || python -c "import json,numpy as np; from sklearn.model_selection import StratifiedKFold; \
y=np.load('data/kvasir224.npz')['y']; json.dump([[a.tolist(),b.tolist()] for a,b in \
StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y)],open('results/folds.json','w'))"

# 3. Features: frozen ImageNet NASNet-Mobile [~5 min] and NASNet-Mobile fine-tuned inside each training fold [~80 min]
[ -f data/feats.npy ] || python -m feature_extraction.features data/kvasir224.npz data/feats.npy
python -m feature_extraction.finetune_features data/kvasir224.npz results/folds.json 5

# 4. End-to-end baselines on the same folds: ResNet-50, EfficientNet-B0, Swin-T  [~75 min on a GPU]
[ -f results/baselines.json ] || python -m deep_baselines.baselines data/kvasir224.npz results/folds.json results/baselines.json 5 \
    resnet50 efficientnet_b0 swin_tiny_patch4_window7_224

# 5. Experiments 1-3 and PCA on the fine-tuned features, with test-time augmentation  [~25 min]
python -m cross_validation.run_cv_ft data/kvasir224.npz 'data/feats_ft_fold{k}.npy' results_ft "$JOBS"
cp results/baselines.json results_ft/

# 6. Two-stage feature selection (ReliefF -> binary WOA): Experiment 4, ablation, comparison of selection methods,
#    random-search control, frozen-feature control  [~30 min]; search-space and budget analysis  [~30 min]
python -m feature_selection.twostage_pipeline data/kvasir224.npz 'data/feats_ft_fold{k}.npy' data/feats.npy results_twostage "$JOBS"
python -m feature_selection.search_budget data/kvasir224.npz 'data/feats_ft_fold{k}.npy' results_budget 12

# 7. Tables, statistics and figures
python -m analysis.report data/kvasir224.npz results_twostage results_twostage/figs
python -m analysis.twostage_stats data/kvasir224.npz results_twostage/stats.json
python -m figures.fig_convergence results_budget results_twostage results_twostage/figs/Fig12_search_convergence.png
python -m figures.fig1_pipeline results_twostage/figs/Fig1_pipeline.png
[ -f data/bbhe_samples/polyps.jpg ] || { mkdir -p data/bbhe_samples; python -m data_prep.fetch_samples data/tail.bin data/bbhe_samples; }
python -m figures.bbhe_fig data/bbhe_samples results_twostage/figs/Fig2b_BBHE_examples.png
echo "Done: tables in results_twostage/summary.md, statistics in results_twostage/stats.json, figures in results_twostage/figs/"
