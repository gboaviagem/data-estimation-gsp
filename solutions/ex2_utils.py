"""Utility functions for Exercise 2."""
import networkx as nx
import numpy as np
from tqdm import tqdm


def adj_matrix_from_coords_limited(coords, limit):
    """Create a nearest-neighbors graph with gaussian weights.

    Parameters
    ----------
    coords : array
        (N, 2) array of coordinates.
    limit : int
        Minimum number of neighbors.

    References
    ----------
    SHUMAN, D. I. et al. The emerging field of signal processing on graphs:
    Extending high-dimensional data analysis to networks and other irregular
    domains. IEEE Signal Process. Mag., IEEE, v. 30, n. 3, p. 83–98, 2013

    """
    [N, _] = coords.shape
    A = np.zeros((N,N))

    for	i in tqdm(np.arange(1, N)):
        dist2i = np.sqrt(
            (coords[:,0] - coords[i,0]) ** 2 +
            (coords[:,1] - coords[i,1]) ** 2)

        idx = np.argsort(dist2i)[1:limit+1]
        for j in idx:
            distance = np.linalg.norm(coords[i, :] - coords[j, :], ord=2)
            if A[i, j] == 0:
                A[i, j] = np.exp(-(distance**2))

    return A + A.transpose()


def describe(graph, return_dict=False):
    """Compute and print some graph-related metrics.

    Parameters
    ----------
    graph : nx.Graph() or np.ndarray
        If Numpy array, it is assumed to be the adjacency matrix.
    return_dict : bool, optional, default: False
        Whether to return a dictionary with info.
    """
    if isinstance(graph, np.ndarray):
        graph = nx.from_numpy_matrix(graph)

    if isinstance(graph, nx.DiGraph):  # MultiDiGraph is also considered here
        is_dir = True
    else:
        is_dir = False

    d = dict()
    d["n_nodes"] = nx.number_of_nodes(graph)
    d["n_edges"] = nx.number_of_edges(graph)
    d["n_self_loops"] = nx.number_of_selfloops(graph)
    d["density"] = nx.density(graph)
    if is_dir:
        d["is_strongly_connected"] = nx.is_strongly_connected(graph)
        d["n_strongly_connected_components"] = nx.number_strongly_connected_components(graph)
        d["is_weakly_connected"] = nx.is_weakly_connected(graph)
        d["n_weakly_connected_components"] = nx.number_strongly_connected_components(graph)
    else:
        d["is_connected"] = nx.is_connected(graph)
        d["n_connected_components"] = nx.number_connected_components(graph)
    d["is_directed"] = nx.is_directed(graph)
    d["is_weighted"] = nx.is_weighted(graph)

    if return_dict:
        return d
    else:
        for key, value in d.items():
            print(key + ":", value)


def matrix_poly(coeff, M):
    """Polynomial correctly evaluated at a matrix M.

    The order of the input coefficients follows each term degree: first
    coefficient is related to degree 0, and so on.
    """
    p = np.array([
        coeff[i] * np.linalg.matrix_power(M, i)
        for i in range(len(coeff))]).sum(axis=0)
    return p