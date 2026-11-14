"""Step 2: NASNet-Mobile (ImageNet) 'predictions'-layer (pre-softmax logit) features for 5 augmented versions x 2 streams.

Augmentation (rotation and flipping): 0 = original, 1 = rot90, 2 = rot180, 3 = rot270, 4 = horizontal flip.
BBHE commutes with these pixel rearrangements, so augmenting the BBHE stream equals BBHE of augmented images.
Usage: python features.py <data.npz> <feats.npy>
"""
import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf


def augment(x, a):
    if a == 0: return x
    if a in (1, 2, 3): return np.rot90(x, k=a, axes=(1, 2))
    return x[:, :, ::-1]


def main(data_path, out):
    d = np.load(data_path)
    full = tf.keras.applications.NASNetMobile(weights='imagenet')
    pred = full.get_layer('predictions')            # Keras folds the softmax into this layer
    model = tf.keras.Model(full.input, pred.input)   # 1056-D global-average-pooling output
    W, b = pred.get_weights()                        # logits = GAP @ W + b (MATLAB 'predictions' FC layer)
    feats = np.zeros((2, 5, len(d['y']), 1000), dtype=np.float32)
    t0 = time.time()
    for s, key in enumerate(('orig', 'bbhe')):
        imgs = d[key]
        for a in range(5):
            for i in range(0, len(imgs), 256):
                x = augment(imgs[i:i + 256], a).astype(np.float32)
                x = tf.keras.applications.nasnet.preprocess_input(x)
                feats[s, a, i:i + 256] = model.predict(x, batch_size=64, verbose=0) @ W + b
            print(f'stream {key} aug {a} done ({time.time()-t0:.0f}s)', flush=True)
    np.save(out, feats)
    print('saved', out, feats.shape, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
