"""Figure 1: overview of the pipeline and of the evaluation protocol (training fold vs test fold).
Usage: python -m figures.fig1_pipeline <out.png>
"""
import sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = sys.argv[1]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.2})
W, H = 5.46, 6.6
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

TR_FILL, TR_EDGE = '#E8F0FB', '#2E6DB4'      # training branch
TE_FILL, TE_EDGE = '#FDEFE3', '#C8641E'      # test branch
NE_FILL, NE_EDGE = '#F2F2F2', '#555555'      # shared steps
INK = '#1F1F1F'

def box(x, y, w, h, title, sub=None, fill=NE_FILL, edge=NE_EDGE, bold=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                fc=fill, ec=edge, lw=0.9))
    if sub:
        ax.text(x, y + 0.085, title, ha='center', va='center', color=INK, fontweight='bold' if bold else 'normal')
        ax.text(x, y - 0.105, sub, ha='center', va='center', color='#3A3A3A', fontsize=6.3)
    else:
        ax.text(x, y, title, ha='center', va='center', color=INK, fontweight='bold' if bold else 'normal')

def arrow(x1, y1, x2, y2, color=INK, dashed=False, label=None, lx=0, ly=0, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, lw=0.8, color=color,
                                 linestyle=(0, (3, 2)) if dashed else '-', connectionstyle=f'arc3,rad={rad}'))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha='center', va='center', fontsize=6.1,
                color=color, style='italic', bbox=dict(fc='white', ec='none', pad=0.6))

# top, shared
cx = W / 2
box(cx, 6.27, 3.3, 0.42, 'Kvasir v2: 8,000 images, 8 classes (1,000 each)', 'resized to 224 × 224')
box(cx, 5.62, 3.3, 0.42, 'Stratified 5-fold split on the original images', 'repeated for each fold k = 1, …, 5')
arrow(cx, 6.06, cx, 5.83)

# lanes
LX, RX, BW = 1.24, 4.22, 2.24
ax.text(LX, 5.18, 'TRAINING FOLD (6,400 images)', ha='center', fontweight='bold', color=TR_EDGE, fontsize=7.4)
ax.text(RX, 5.18, 'TEST FOLD (1,600 images)', ha='center', fontweight='bold', color=TE_EDGE, fontsize=7.4)
arrow(cx - 0.5, 5.41, LX, 5.27, color=TR_EDGE); arrow(cx + 0.5, 5.41, RX, 5.27, color=TE_EDGE)

ys = [4.82, 4.24, 3.66, 3.08, 2.50, 1.92, 1.34]
tr = [('Augmentation', 'original + rotations 90°/180°/270° + flip (×5)'),
      ('BBHE enhancement', 'Y channel; original and BBHE streams'),
      ('Fine-tune NASNet-Mobile', 'ImageNet init; 5 epochs; mixed batches'),
      ('Pooled features', '1,056-D per stream (global average pooling)'),
      ('Serial fusion', '2,112-D vector per image'),
      ('Two-stage feature selection', 'ReliefF top 200 → binary WOA (50 × 100)'),
      ('Train SVMs', 'L, Q, CB, MG, CG kernels')]
te = [('Five versions per image', 'used only for test-time augmentation'),
      ('BBHE enhancement', 'same as training'),
      ('Fine-tuned NASNet-Mobile', 'forward pass only (no training)'),
      ('Pooled features', '1,056-D per stream'),
      ('Serial fusion', '2,112-D vector per image'),
      ('Apply feature mask', 'selected on the training fold'),
      ('Trained SVM + TTA', 'decision scores averaged over 5 versions')]
for i, y in enumerate(ys):
    box(LX, y, BW, 0.44, *tr[i], fill=TR_FILL, edge=TR_EDGE)
    box(RX, y, BW, 0.44, *te[i], fill=TE_FILL, edge=TE_EDGE)
    if i:
        arrow(LX, ys[i - 1] - 0.22, LX, y + 0.22, color=TR_EDGE)
        arrow(RX, ys[i - 1] - 0.22, RX, y + 0.22, color=TE_EDGE)

# what passes from training to test (learned once on the training fold)
mid = (LX + RX) / 2
for i, lab in ((2, 'weights'), (5, 'feature mask'), (6, 'classifier')):
    arrow(LX + BW / 2, ys[i], RX - BW / 2, ys[i], color='#555555', dashed=True, label=lab, ly=0.0)

# output
box(RX, 0.62, BW, 0.42, 'Predicted class (8 classes)', 'pooled over 5 folds: accuracy, macro-F1', fill=NE_FILL, edge=NE_EDGE)
arrow(RX, ys[-1] - 0.22, RX, 0.83, color=TE_EDGE)
ax.text(LX, 0.62, 'Test images never enter augmentation,\nfine-tuning, feature selection or SVM training.',
        ha='center', va='center', fontsize=6.3, color='#3A3A3A', style='italic')
fig.savefig(out)
print('saved', out)
