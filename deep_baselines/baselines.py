"""Step 4: end-to-end CNN / Transformer baselines on the SAME folds (Apple GPU via PyTorch MPS).

Each model is fine-tuned from ImageNet weights on the training originals of a fold with on-the-fly rotation/flip
augmentation (the same five transforms as the proposed pipeline), then evaluated on the non-augmented test fold.
Usage: python -m deep_baselines.baselines <data.npz> <folds.json> <out.json> <epochs> <model> [<model> ...]
"""
import json, sys, time
import numpy as np
import timm, torch
import torch.nn.functional as Fn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def to_tensor(x):
    return ((torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255) - MEAN) / STD


def augment(x, rng):
    a = rng.integers(5)
    if a == 4: return torch.flip(x, dims=[3])
    return torch.rot90(x, k=int(a), dims=[2, 3])


def run(model_name, imgs, y, folds, epochs, bs=48):
    res = []
    for k, (tr, te) in enumerate(folds):
        tr, te = np.array(tr), np.array(te); rng = np.random.default_rng(k)
        torch.manual_seed(k)
        m = timm.create_model(model_name, pretrained=True, num_classes=8).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.05)
        steps = epochs * int(np.ceil(len(tr) / bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
        t0 = time.perf_counter(); m.train()
        for ep in range(epochs):
            perm = rng.permutation(tr)
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                x = augment(to_tensor(imgs[b]), rng); t = torch.from_numpy(y[b]).to(dev)
                loss = Fn.cross_entropy(m(x), t, label_smoothing=0.1)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        torch.mps.synchronize() if dev.type == 'mps' else None
        train_s = time.perf_counter() - t0
        m.eval(); preds = []; t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(te), 128):
                preds.append(m(to_tensor(imgs[te[i:i + 128]])).argmax(1).cpu().numpy())
        pred_s = time.perf_counter() - t0; pred = np.concatenate(preds)
        p, r, f1, _ = precision_recall_fscore_support(y[te], pred, average='macro', zero_division=0)
        res.append(dict(model=model_name, fold=k, acc=float(accuracy_score(y[te], pred)), precision=float(p), recall=float(r),
                        f1=float(f1), train_s=train_s, pred_speed=len(te) / pred_s, epochs=epochs,
                        params_m=sum(q.numel() for q in m.parameters()) / 1e6, pred=pred.tolist(), te=te.tolist()))
        print(model_name, 'fold', k, f"acc={res[-1]['acc']:.4f}", f'train={train_s:.0f}s', flush=True)
        del m, opt
        if dev.type == 'mps': torch.mps.empty_cache()
    return res


def main(data_path, folds_path, out, epochs, *models):
    d = np.load(data_path); imgs, y = d['orig'], d['y']; folds = json.load(open(folds_path))
    allres = []
    for name in models:
        allres += run(name, imgs, y, folds, int(epochs))
        json.dump(allres, open(out, 'w'))


if __name__ == '__main__':
    main(*sys.argv[1:])
