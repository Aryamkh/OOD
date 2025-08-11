import numpy as np, torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

def score_msp(logits):
    return 1.0 - F.softmax(logits, dim=1).max(dim=1).values

def score_maxlogit(logits):
    return -logits.max(dim=1).values

def score_energy(logits, T=1.0):
    E = -T * torch.logsumexp(logits / T, dim=1)
    return -E

def odin_pre_norm_score(model, loader, device, eps, T, mean, std):
    m = torch.tensor(mean, device=device)[None, :, None, None]
    s = torch.tensor(std,  device=device)[None, :, None, None]
    scores = []
    model.eval()
    for xb, _ in loader:
        xn = xb.to(device, non_blocking=True)
        x  = xn * s + m
        x  = x.detach().clone().requires_grad_(True)

        logits = model((x - m) / s)
        preds = logits.argmax(1)
        loss  = F.cross_entropy(logits / T, preds)
        loss.backward()
        x_adv = torch.clamp(x + eps * x.grad.sign(), 0.0, 1.0)
        with torch.no_grad():
            logits_adv = model((x_adv - m) / s)
            E = -T * torch.logsumexp(logits_adv / T, dim=1)
            scores.append(-E.detach().cpu())
    return torch.cat(scores)

def react_fit_percentile(id_feats, percent=0.9):
    thr = torch.quantile(id_feats, q=percent, dim=0)
    return thr

def react_apply(feats, thr):
    return torch.minimum(feats, thr)

def ash_apply(feats, p=0.9):
    B, D = feats.shape
    k = int(p * D)
    if k <= 0: return feats
    out = feats.clone()
    vals, idx = torch.topk(out.abs(), k=k, dim=1, largest=True, sorted=False)
    mask = torch.zeros_like(out, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    out[mask] = 0.0
    pre_norm = feats.norm(dim=1, keepdim=True) + 1e-6
    post_norm = out.norm(dim=1, keepdim=True) + 1e-6
    out = out * (pre_norm / post_norm)
    return out

def vim_fit(id_feats, id_logits):
    mu = id_feats.mean(0, keepdim=True)
    R = id_feats - mu
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    v = Vh[0]
    return mu, v

def vim_score(id_mu, v, feats, logits, alpha=0.05):
    r = feats - id_mu
    b = (r @ v)
    maxlogit = logits.max(1).values
    return b - alpha * maxlogit

def fit_mahalanobis(train_feats, train_labels, num_classes):
    means = []
    for c in range(num_classes):
        means.append(train_feats[train_labels==c].mean(0, keepdim=True))
    means = torch.cat(means, 0)
    centered = []
    for c in range(num_classes):
        f = train_feats[train_labels==c] - means[c]
        centered.append(f)
    centered = torch.cat(centered, 0)
    cov = torch.from_numpy(np.cov(centered.numpy().T)).float()
    cov += 1e-5*torch.eye(cov.size(0))
    inv_cov = torch.inverse(cov)
    return means, inv_cov

def score_mahalanobis(feats, means, inv_cov):
    diffs = feats[:, None, :] - means[None, :, :]
    M = torch.einsum("bcd,dd,bce->bc", diffs, inv_cov, diffs)
    return torch.min(M, dim=1).values

def fit_knn(train_feats, n_neighbors=50):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="euclidean")
    nbrs.fit(train_feats.numpy())
    return nbrs

def score_knn(nbrs, feats, k=10):
    dists, _ = nbrs.kneighbors(feats.numpy(), n_neighbors=k, return_distance=True)
    return torch.from_numpy(dists[:, -1])

def adascale_score(feats, logits, gamma=0.2):
    q = torch.quantile(feats.abs(), q=0.2, dim=1) + 1e-6
    m = feats.abs().mean(dim=1) + 1e-6
    r = q / m
    scaled = logits / (1.0 + gamma * r[:, None])
    E = -torch.logsumexp(scaled, dim=1)
    return -E
