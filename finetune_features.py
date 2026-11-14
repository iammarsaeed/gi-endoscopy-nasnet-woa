"""Option 1+2: fine-tune NASNet-Mobile inside each training fold, then extract 1056-D pooled features.

For fold k: the network (ImageNet weights, new 8-class head) is fine-tuned on the TRAINING images of fold k only;
each batch mixes original and BBHE images (50/50) with random rotation by 0/90/180/270 degrees or horizontal flip.
Then pooled features are extracted for all 8,000 images x 5 versions x 2 streams with that fold's network
(test images are only passed forward, never used for training). Saves data/feats_ft_fold{k}.npy (2, 5, 8000, 1056).
Usage: python finetune_features.py <data.npz> <folds.json> <epochs> [fold ...]
"""
import json, os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from features import augment

BS = 32


def build():
    base = tf.keras.applications.NASNetMobile(weights='imagenet', include_top=False, pooling='avg',
                                              input_shape=(224, 224, 3))
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(8, activation='softmax')(x)
    return tf.keras.Model(base.input, out), base


def batches(orig, bbhe, y, idx, rng):
    idx = rng.permutation(idx)
    for i in range(0, len(idx) - BS + 1, BS):
        b = np.sort(idx[i:i + BS])
        use_b = rng.random(len(b)) < 0.5
        x = np.where(use_b[:, None, None, None], bbhe[b], orig[b])
        x = np.stack([augment(im[None], int(a))[0] for im, a in zip(x, rng.integers(5, size=len(b)))])
        yield tf.keras.applications.nasnet.preprocess_input(x.astype(np.float32)), y[b]


def main(data_path, folds_path, epochs, *which):
    d = np.load(data_path); orig, bbhe, y = d['orig'], d['bbhe'], d['y']
    folds = json.load(open(folds_path)); epochs = int(epochs)
    todo = [int(k) for k in which] if which else range(5)
    for k in todo:
        out = f'data/feats_ft_fold{k}.npy'
        if os.path.exists(out): continue
        tr = np.array(folds[k][0]); rng = np.random.default_rng(100 + k); tf.keras.utils.set_random_seed(100 + k)
        model, base = build()
        steps = epochs * (len(tr) // BS)
        lr = tf.keras.optimizers.schedules.CosineDecay(1e-4, decay_steps=steps, warmup_target=3e-4, warmup_steps=steps // 10)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        t0 = time.time()
        for ep in range(epochs):
            accs = []
            for x, t in batches(orig, bbhe, y, tr, rng):
                _, acc = model.train_on_batch(x, t); accs.append(acc)
            print(f'fold {k} epoch {ep + 1}/{epochs} train acc {np.mean(accs):.3f} ({time.time() - t0:.0f}s)', flush=True)
        train_s = time.time() - t0
        feats = np.zeros((2, 5, len(y), 1056), dtype=np.float32); t1 = time.time()
        for s, imgs in enumerate((orig, bbhe)):
            for a in range(5):
                for i in range(0, len(imgs), 256):
                    x = tf.keras.applications.nasnet.preprocess_input(augment(imgs[i:i + 256], a).astype(np.float32))
                    feats[s, a, i:i + 256] = base.predict(x, batch_size=128, verbose=0)
        np.save(out, feats)
        json.dump(dict(fold=k, epochs=epochs, finetune_s=train_s, extract_s=time.time() - t1),
                  open(f'data/feats_ft_fold{k}.json', 'w'))
        print(f'fold {k} done: finetune {train_s:.0f}s, extract {time.time() - t1:.0f}s', flush=True)
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main(*sys.argv[1:])
