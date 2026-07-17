"""
Discrete curve smoothing via mean curvature flow (curve shortening flow).

A chapter's paragraphs, connected in story order, form a noisy polyline in
embedding space. Mean curvature flow moves each point along the curve's
curvature normal, which shortens and smooths the polyline while keeping it
close to the original data (the discrete analogue of curve-shortening flow
for triangle-mesh surfaces, see DDG lecture notes on discrete curves).

The discrete curvature normal at interior vertex i of an open polyline is
the length-weighted Laplacian of the vertex positions:

    (kappa N)_i = 2 / (l_{i-1} + l_i) * ( (x_{i+1} - x_i) / l_i - (x_i - x_{i-1}) / l_{i-1} )

where l_i = |x_{i+1} - x_i| is the length of edge i. Flowing dx/dt = (kappa N)
shortens the curve; we integrate it with implicit (backward Euler) time
stepping, which is unconditionally stable and lets us take large steps,
unlike naive explicit Laplacian smoothing (see page 33 of the MIT DDG notes
on mean curvature flow for triangle meshes -- the same implicit trick
applies here).

Endpoints are pinned (Dirichlet boundary conditions): a chapter's first and
last paragraph are the fixed anchors of its arc, and pinning them stops the
flow from shrinking the whole curve to a point.

Running the flow directly on the paragraph points is not enough to get a
curved result: with only 3 vertices and both endpoints pinned, the middle
vertex has nowhere meaningful to go, and paragraph-to-paragraph edges that
are long relative to their neighbors stay almost straight after flowing --
the flow only has as many degrees of freedom as there are original points.
The fix (per Congyue Deng's guidance) is to subsample many extra control
points along the original polyline *before* flowing, uniformly in arc
length so longer edges -- which need more room to bend -- automatically
receive more control points than short ones. `uniform_arclength_resample`
does this subsampling; `mean_curvature_flow` should then be run on the
dense resampled polyline rather than the sparse original one.

Example:
    from geometry.curve_flow import uniform_arclength_resample, mean_curvature_flow
    dense, orig_idx = uniform_arclength_resample(coords, samples_per_edge=10)
    smoothed = mean_curvature_flow(dense, iterations=5, step_size=0.15)
    smoothed_at_paragraphs = smoothed[orig_idx]
"""

import numpy as np
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import splu


def uniform_arclength_resample(points, samples_per_edge=10, min_samples=None):
    """
    Insert extra control points along an open polyline, uniformly spaced in
    arc length across the whole curve.

    Sampling uniformly in arc length -- rather than a fixed number of extra
    points per edge -- means long edges (large gaps between paragraphs)
    automatically get proportionally more control points than short ones,
    since they span more of the total arc length. This is what gives mean
    curvature flow enough freedom to bend a long jump into a curve instead
    of leaving it as a near-straight segment.

    Args:
        points:           (n, d) original vertex positions, in curve order
        samples_per_edge: target average number of output samples per
                           original edge; total output size is
                           ~ (n - 1) * samples_per_edge
        min_samples:      floor on total output points (defaults to n,
                           i.e. never return fewer points than the input)

    Returns:
        dense:    (m, d) resampled points, m >= n
        orig_idx: (n,) index into `dense` of each original input point
                  (the closest dense sample to that point's arc-length
                  position), so callers can locate/highlight the original
                  paragraphs on the smoothed curve
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        return pts.copy(), np.arange(n)

    seg_vec = pts[1:] - pts[:-1]
    seg_len = np.maximum(np.linalg.norm(seg_vec, axis=1), 1e-12)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]

    n_samples = max(n, (n - 1) * samples_per_edge + 1)
    if min_samples is not None:
        n_samples = max(n_samples, min_samples)

    if total < 1e-12:
        return pts.copy(), np.arange(n)

    target_s = np.linspace(0.0, total, n_samples)
    dense = np.empty((n_samples, pts.shape[1]))
    for d in range(pts.shape[1]):
        dense[:, d] = np.interp(target_s, cum, pts[:, d])

    orig_idx = np.searchsorted(target_s, cum)
    orig_idx = np.clip(orig_idx, 0, n_samples - 1)
    return dense, orig_idx


def curve_laplacian(points):
    """
    Build the length-weighted Laplacian of an open polyline.

    Boundary rows (0 and n-1) are left as all-zero; callers pin those rows
    to identity when they want fixed endpoints.

    Args:
        points: (n, d) vertex positions

    Returns:
        L: (n, n) sparse matrix, L @ points ~= curvature normal at each vertex
    """
    n = len(points)
    edge_vec = points[1:] - points[:-1]
    edge_len = np.linalg.norm(edge_vec, axis=1)
    edge_len = np.maximum(edge_len, 1e-10)

    idx = np.arange(1, n - 1)
    l_prev = edge_len[idx - 1]
    l_next = edge_len[idx]
    w_prev = 2.0 / (l_prev * (l_prev + l_next))
    w_next = 2.0 / (l_next * (l_prev + l_next))

    rows = np.concatenate([idx, idx, idx])
    cols = np.concatenate([idx - 1, idx, idx + 1])
    vals = np.concatenate([w_prev, -(w_prev + w_next), w_next])

    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def mean_curvature_flow(points, iterations=5, step_size=0.15):
    """
    Smooth an open polyline by implicit mean curvature flow.

    Each step solves (I - h*L) x_new = x_old for a fresh Laplacian L built
    from the current geometry (semi-implicit: linear per step, still
    nonlinear -- and stable -- across steps). curve_laplacian only defines
    rows for interior vertices (1..n-2); its boundary rows are zero, so the
    endpoints are structurally pinned to their original positions by every
    step of this flow -- there's no free-boundary variant.

    The curvature-normal weights in L scale as 1/length^2, so a fixed h
    would smooth tightly-clustered chapters into a straight line almost
    immediately while barely touching spread-out ones. To keep "amount of
    smoothing per step" comparable across curves of very different scale,
    h is rescaled each iteration by the curve's own median squared edge
    length: h_eff = step_size * median(edge_length)^2. step_size is then a
    small dimensionless fraction (roughly 0-1) of the curve's local scale
    per step, not an absolute distance.

    Note that with fixed endpoints and enough total flow time, curve
    shortening flow's only stationary point is the straight segment between
    them -- that's the correct limit, just not a useful amount of denoising.
    Keep iterations low; this is meant to remove local noise, not flatten
    the chapter's shape.

    Args:
        points:      (n, d) vertex positions (any embedding dimension)
        iterations:  number of implicit Euler steps
        step_size:   dimensionless flow amount per step (auto-scaled by
                     the curve's own median squared edge length)

    Returns:
        (n, d) smoothed vertex positions
    """
    X = np.asarray(points, dtype=float).copy()
    n = len(X)
    if n < 3:
        return X

    x0, x_end = X[0].copy(), X[-1].copy()

    for _ in range(iterations):
        edge_len = np.linalg.norm(X[1:] - X[:-1], axis=1)
        scale = np.median(edge_len) ** 2
        h = step_size * max(scale, 1e-10)

        L = curve_laplacian(X)
        A = (eye(n, format="lil") - h * L).tolil()
        A.rows[0], A.data[0] = [0], [1.0]
        A.rows[-1], A.data[-1] = [n - 1], [1.0]

        rhs = X.copy()
        rhs[0], rhs[-1] = x0, x_end

        solver = splu(A.tocsc())
        X = solver.solve(rhs)

    return X


def total_curvature(points):
    """Sum of turning angles (radians) along a polyline -- a roughness score."""
    if len(points) < 3:
        return 0.0
    v1 = points[1:-1] - points[:-2]
    v2 = points[2:] - points[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    safe = (n1 > 1e-10) & (n2 > 1e-10)
    cos_angle = np.ones(len(v1))
    cos_angle[safe] = np.clip(
        (v1[safe] * v2[safe]).sum(axis=1) / (n1[safe] * n2[safe]), -1.0, 1.0
    )
    return float(np.arccos(cos_angle).sum())
