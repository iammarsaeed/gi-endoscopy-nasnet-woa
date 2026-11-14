"""Step 1: extract Kvasir v2, verify every image against the official archive's CRC32 index,
resize to 224x224, and build the BBHE stream.

Usage: python prep.py <mirror.zip> <official_tail.bin> <out.npz>
"""
import io, os, struct, sys, zipfile, zlib
from multiprocessing import Pool
import numpy as np
from PIL import Image

OFFICIAL_TOTAL = 2489312085
CLASSES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
           'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']


def official_index(tail_path):
    tail = open(tail_path, 'rb').read(); base = OFFICIAL_TOTAL - len(tail)
    eocd = tail.rfind(b'PK\x05\x06')
    n, _, off = struct.unpack('<HII', tail[eocd + 10:eocd + 20])
    pos = off - base; idx = {}
    for _ in range(n):
        f = struct.unpack('<IHHHHHHIIIHHHHHII', tail[pos:pos + 46])
        crc, nlen, xlen, clen = f[7], f[10], f[11], f[12]
        name = tail[pos + 46:pos + 46 + nlen].decode('utf8')
        if name.endswith('.jpg'):
            idx['/'.join(name.split('/')[-2:])] = crc
        pos += 46 + nlen + xlen + clen
    return idx


def bbhe(y):
    """Brightness-preserving bi-histogram equalization (Kim, 1997) of a uint8 channel."""
    y = y.astype(np.int64); xm = int(np.round(y.mean())); out = np.empty_like(y)
    for mask, a, b in ((y <= xm, 0, xm), (y > xm, xm + 1, 255)):
        v = y[mask]
        if v.size == 0 or b <= a:
            out[mask] = v; continue
        cdf = np.cumsum(np.bincount(v - a, minlength=b - a + 1)) / v.size
        out[mask] = np.round(a + (b - a) * cdf[v - a]).astype(np.int64)
    return out.astype(np.uint8)


def load(args):
    key, data = args
    im = Image.open(io.BytesIO(data)).convert('RGB').resize((224, 224), Image.BILINEAR)
    rgb = np.asarray(im, dtype=np.uint8)
    ycc = np.array(im.convert('YCbCr')); ycc[..., 0] = bbhe(ycc[..., 0])
    enh = np.asarray(Image.fromarray(ycc, 'YCbCr').convert('RGB'), dtype=np.uint8)
    return key, rgb, enh


def main(zip_path, tail_path, out):
    idx = official_index(tail_path)
    assert len(idx) == 8000, len(idx)
    items = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if not info.filename.endswith('.jpg'):
                continue
            key = '/'.join(info.filename.split('/')[-2:])
            if key.split('/')[0] not in CLASSES:
                continue
            data = z.read(info)
            if zlib.crc32(data) != idx.get(key):
                raise SystemExit(f'CRC mismatch or unknown file: {key}')
            items.append((key, data))
    keys = sorted(k for k, _ in items)
    assert len(keys) == 8000 and set(keys) == set(idx), 'file set differs from official archive'
    print('verified 8000/8000 images against the official Kvasir v2 CRC32 index', flush=True)
    items.sort()
    with Pool(os.cpu_count()) as p:
        res = p.map(load, items, chunksize=50)
    names = np.array([k for k, _, _ in res])
    y = np.array([CLASSES.index(k.split('/')[0]) for k in names], dtype=np.int64)
    np.savez(out, orig=np.stack([r[1] for r in res]), bbhe=np.stack([r[2] for r in res]), y=y, names=names)
    print('saved', out, np.bincount(y), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
