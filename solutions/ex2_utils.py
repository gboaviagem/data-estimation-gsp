"""Utility functions for Exercise 2."""
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np


def nearest_neighbors(X, n_neighbors=5, algorithm='ball_tree',
                      mode='distance'):
    """Return the nearest neighbors' graph weighted adjacency matrix.

    This is a wrapper for the Scikit-learn NearestNeighbors.kneighbors_graph
    method.

    Parameters
    ----------
    X : np.ndarray
        Input data 2D array.
    n_neighbors : int, default=5
        Number of neighbors to use by default for kneighbors queries.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

            'ball_tree' will use BallTree

            'kd_tree' will use KDTree

            'brute' will use a brute-force search.

            'auto' will attempt to decide the most appropriate algorithm
            based on the values passed to fit method.

    mode : {'connectivity', 'distance'}, default='connectivity'
        Type of returned matrix: `connectivity` will return the connectivity
        matrix with ones and zeros, and `distance` will return the distances
        between neighbors according to the given metric.

	Return
	------
	W : weighted adjacency matrix in CSR (Compressed Sparse Row) format
    """
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm=algorithm).fit(X)
    W = nbrs.kneighbors_graph(X, mode=mode)

    return W


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
