"""
KNN graph construction and geodesic distances on the story manifold.

A story manifold is approximated by building a k-nearest-neighbour (KNN)
graph over paragraph embeddings. Geodesics are shortest paths on this graph
computed with Dijkstra's algorithm.

Example:
    from geometry.knn import build_knn_graph, compute_geodesics, geodesic_path
    graph, knn_indices, knn_distances = build_knn_graph(embeddings, k=10)
    dist_matrix = compute_geodesics(graph)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def build_knn_graph(embeddings, k=10, metric="cosine"):
    """
    Build a symmetric KNN graph from paragraph embeddings.

    Each paragraph is connected to its k nearest neighbours. The graph is
    symmetrised so that if paragraph i is near j, j is also connected to i
    (union of directed edges). Edge weights are distances (smaller = closer).

    Args:
        embeddings:     (N, D) float array of L2-normalised paragraph embeddings
        k:              number of nearest neighbours per node
        metric:         'cosine' (default) or 'euclidean'

    Returns:
        graph:          (N, N) sparse CSR matrix; graph[i, j] = distance, 0 if no edge
        knn_indices:    (N, k) int array of neighbour indices
        knn_distances:  (N, k) float array of distances to those neighbours
    """
    n = len(embeddings)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Drop the first column (each point is its own nearest neighbour at distance 0)
    knn_distances = distances[:, 1:]
    knn_indices = indices[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = knn_indices.ravel()
    weights = knn_distances.ravel()

    adj = csr_matrix((weights, (rows, cols)), shape=(n, n))

    # Symmetrise: cosine distance is symmetric, so d(i,j) == d(j,i).
    # adj.maximum(adj.T) keeps the maximum at each position, which for an
    # asymmetric binary pattern means "keep d wherever *either* direction
    # has an edge, and use the actual distance value in both cells".
    graph = adj.maximum(adj.T)

    return graph, knn_indices, knn_distances


def compute_geodesics(graph, source=None):
    """
    Compute geodesic distances on the KNN graph via Dijkstra's algorithm.

    Geodesic distance is the length of the shortest path along graph edges,
    which approximates the true geodesic distance on the underlying manifold.

    Args:
        graph:    (N, N) sparse CSR matrix from build_knn_graph
        source:   None  -> return full (N, N) all-pairs distance matrix
                  int   -> return (N,) distances from that single source
                  list  -> return (|source|, N) distances from each source

    Returns:
        dist_matrix: array of geodesic distances; np.inf where no path exists
    """
    return dijkstra(graph, directed=False, indices=source, unweighted=False)


def geodesic_path(graph, source, target):
    """
    Return the paragraph indices along the shortest path from source to target.

    Args:
        graph:   (N, N) sparse CSR matrix from build_knn_graph
        source:  int, start paragraph index
        target:  int, end paragraph index

    Returns:
        path: list of paragraph indices from source to target (inclusive),
              or [] if no path exists
    """
    _, predecessors = dijkstra(
        graph,
        directed=False,
        indices=source,
        return_predecessors=True,
    )

    path = []
    node = target
    while node != source and node >= 0:
        path.append(int(node))
        node = int(predecessors[node])

    if node == source:
        path.append(source)
        path.reverse()
        return path

    return []  # disconnected graph — no path exists
