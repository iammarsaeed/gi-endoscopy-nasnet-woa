import sys, os
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def bbhe(y):
    """Brightness-preserving bi-histogram equalization (Kim, 1997) of a uint8 channel."""
    y = y.astype(np.int64); xm = int(np.round(y.mean()))
    out = np.empty_like(y)
    lo, hi = y <= xm, y > xm
    for mask, a, b in ((lo, 0, xm), (hi, xm + 1, 255)):
        v = y[mask]
        if v.size == 0 or b <= a:
            out[mask] = v; continue
        hist = np.bincount(v - a, minlength=b - a + 1).astype(float)
        cdf = np.cumsum(hist) / v.size
        out[mask] = np.round(a + (b - a) * cdf[v - a]).astype(np.int64)
    return out.astype(np.uint8)

src, dst = sys.argv[1], sys.argv[2]
names = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
         'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8})
fig, ax = plt.subplots(3, 8, figsize=(12, 4.6), dpi=200, gridspec_kw={'height_ratios': [1, 1, 0.8]})
stats = []
for j, n in enumerate(names):
    im = Image.open(os.path.join(src, n + '.jpg')).convert('RGB').resize((224, 224))
    ycc = np.array(im.convert('YCbCr'))
    y0 = ycc[..., 0].copy(); y1 = bbhe(y0); ycc[..., 0] = y1
    out = Image.fromarray(ycc, 'YCbCr').convert('RGB')
    ax[0, j].imshow(im); ax[1, j].imshow(out)
    ax[0, j].set_title(n.replace('-', ' ').capitalize(), fontsize=7.5)
    for r in (0, 1): ax[r, j].axis('off')
    ax[2, j].hist(y0.ravel(), bins=64, range=(0, 255), color='#999999', alpha=0.8, label='Original')
    ax[2, j].hist(y1.ravel(), bins=64, range=(0, 255), color='#2E6DB4', alpha=0.6, label='BBHE')
    ax[2, j].set_yticks([]); ax[2, j].set_xticks([0, 128, 255]); ax[2, j].tick_params(labelsize=6)
    stats.append((n, y0.mean(), y1.mean(), y0.std(), y1.std()))
ax[0, 0].text(-0.08, 0.5, 'Original', transform=ax[0, 0].transAxes, rotation=90, va='center', ha='right', fontsize=8)
ax[1, 0].text(-0.08, 0.5, 'BBHE', transform=ax[1, 0].transAxes, rotation=90, va='center', ha='right', fontsize=8)
ax[2, 0].text(-0.08, 0.5, 'Y histogram', transform=ax[2, 0].transAxes, rotation=90, va='center', ha='right', fontsize=8)
ax[2, 7].legend(fontsize=6, frameon=False, loc='upper right')
fig.tight_layout(); fig.savefig(dst); plt.close(fig)
for s in stats: print(f'{s[0]:24s} mean Y {s[1]:6.1f} -> {s[2]:6.1f}   std {s[3]:5.1f} -> {s[4]:5.1f}')
