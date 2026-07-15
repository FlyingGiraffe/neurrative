"""
Discrete Laplacian operator and spectral decomposition on the story manifold.

The graph Laplacian L = D - W generalises the Laplace-Beltrami operator to
discrete point clouds. Its eigenvectors are the "Fourier modes" of the manifold:
low-frequency modes capture global structure (overall narrative arc), and
high-frequency modes capture local variation between nearby paragraphs.

Example:
    from geometry.knn import build_knn_graph
    from geometry.laplacian import build_laplacian, spectral_decomposition

    graph, _, _ = build_knn_graph(embeddings, k=10)
    L, degree = build_laplacian(graph)
    eigenvalues, eigenvectors = spectral_decomposition(L, k=20)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def build_laplacian(graph, normalized=True):
    """
    Build the graph Laplacian from a KNN adjacency matrix.

    Raw edge weights are distances (larger = farther apart). We first convert
    them to similarities using a heat kernel:  w_ij = exp(-d_ij^2 / sigma^2),
    where sigma is the median distance across all edges. This makes nearby
    paragraphs strongly connected and distant ones weakly connected.

    Two variants are available:
      Unnormalized:  L = D - W
      Normalized:    L_sym = D^{-1/2} L D^{-1/2}  (recommended; scale-invariant)

    Args:
        graph:       (N, N) sparse CSR matrix from build_knn_graph
        normalized:  if True, return the symmetric normalised Laplacian

    Returns:
        L:      (N, N) sparse Laplacian matrix
        degree: (N,) degree vector — sum of similarity weights per node
    """
    W = graph.copy().astype(float)

    # Convert distances to similarities via heat kernel
    sigma = np.median(W.data)
    W.data = np.exp(-(W.data ** 2) / (sigma ** 2))

    degree = np.array(W.sum(axis=1)).ravel()
    D = diags(degree)
    L = D - W

    if normalized:
        inv_sqrt_deg = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        D_inv_sqrt = diags(inv_sqrt_deg)
        L = D_inv_sqrt @ L @ D_inv_sqrt

    return L, degree


def spectral_decomposition(L, k=20):
    """
    Compute the k smallest eigenpairs of the Laplacian.

    The smallest eigenvalue is always 0 (the constant function is in the null
    space of L). The remaining eigenvectors reveal the shape of the manifold:
    - Eigenvector 1 often separates the beginning from the end of the story.
    - Eigenvector 2 separates subplots or character arcs.
    - Higher eigenvectors capture finer-grained structure.

    Args:
        L:  (N, N) sparse Laplacian from build_laplacian
        k:  number of eigenpairs to compute (includes the zero eigenpair)

    Returns:
        eigenvalues:   (k,) float array, sorted ascending (eigenvalues[0] ≈ 0)
        eigenvectors:  (N, k) float array; column i is the i-th eigenvector
    """
    eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors


def spectral_embedding(eigenvectors, dims=(1, 2)):
    """
    Project paragraphs onto two selected Laplacian eigenvectors.

    Eigenvector 0 is always constant (all ones) and is skipped by convention.
    Eigenvectors 1 and 2 give the best 2D spectral layout.

    Args:
        eigenvectors: (N, k) array from spectral_decomposition
        dims:         tuple of two column indices to use as axes

    Returns:
        coords: (N, 2) 2D coordinates in spectral space
    """
    return eigenvectors[:, list(dims)]
