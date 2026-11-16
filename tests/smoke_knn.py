"""Smoke test: RSFK approximate K-NNG against a brute-force K-NN on the GPU.

Fails if the library returns invalid indices, wrong distances, or a recall
that is far below what the algorithm normally reaches on this input.
"""
import sys
import numpy as np
from gpu_rsfk.RSFK import RSFK

N, D, K = 1000, 16, 8

rng = np.random.RandomState(0)
points = rng.random_sample((N, D))
pts32 = points.astype(np.float32)

# brute-force reference (squared L2)
sq = (pts32 ** 2).sum(1)
d2 = sq[:, None] + sq[None, :] - 2.0 * pts32 @ pts32.T
ref = np.argsort(d2, axis=1)[:, :K]

rsfk = RSFK(random_state=0)
idx, dist = rsfk.find_nearest_neighbors(points, K, n_trees=8,
                                        max_tree_children=128,
                                        nn_exploring_factor=2)
idx = np.asarray(idx)
dist = np.asarray(dist)
print("indices", idx.shape, "dist", dist.shape)

errors = []
if idx.shape != (N, K):
    errors.append("unexpected index shape %s" % (idx.shape,))
if (idx < 0).any() or (idx >= N).any():
    errors.append("%d invalid indices" % ((idx < 0) | (idx >= N)).sum())
if not np.isfinite(dist).all():
    errors.append("non-finite distances")
if not errors:
    # each point is its own nearest neighbour (point_in_self_neigh=True)
    self_rate = float((idx[:, 0] == np.arange(N)).mean())
    # reported distances must match the real distances of the returned ids
    real = ((pts32[:, None, :] - pts32[idx]) ** 2).sum(2)
    dist_err = float(np.abs(real - dist).max())
    recall = float(np.mean([len(set(idx[i]) & set(ref[i])) / K for i in range(N)]))
    print("self=%.3f max_dist_err=%.2e recall=%.3f" % (self_rate, dist_err, recall))
    if self_rate < 0.99:
        errors.append("self-neighbour rate %.3f" % self_rate)
    if dist_err > 1e-3:
        errors.append("distance mismatch %.3e" % dist_err)
    if recall < 0.9:
        errors.append("recall %.3f < 0.9" % recall)

if errors:
    print("FAIL: " + "; ".join(errors))
    sys.exit(1)
print("PASS")
