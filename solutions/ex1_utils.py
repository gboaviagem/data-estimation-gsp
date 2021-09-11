"Utility functions for Exercise 1."
import numpy as np
import tqdm
import copy


def adj_matrix_path(N, weights=None, directed=False):
	"""Returns the adjacency matrix of a path graph.

	Parameters
	----------
	N: int
		Number of graph nodes.
	weights: array, default=None
		Array with edge weights. If None, unit weights are used.
		If not None, then the given value of N is replaced by weights+1.
	directed: bool, default=False
		If True, a directed graph is created.

	"""
	if weights is None:
		A = np.tri(N, k=1) - np.tri(N, k=-2) - np.eye(N)
	else:
		N = len(weights) + 1
		A = np.zeros((N, N))
		A[:-1, 1:] = np.diag(weights)
		A = A + A.transpose()
	if directed:
		A = np.tril(A)
	return A

def coords_path(N):
	"""Coordinates of the vertices in the path graph.
	Parameters
	----------
	N : int
		Number of graph vertices.
	"""
	coords = np.array([[i, 0] for i in range(N)])
	return coords

def make_path(N, weights=None, directed=False):
	"""Create adjacency matrix and coordinates of a path graph.

	Parameters
	----------
	N: int
		Number of graph nodes.
	weights: array, default=None
		Array with edge weights. If None, unit weights are used.
		If not None, then the given value of N is replaced by weights+1.
	directed: bool, default=False
		If True, a directed graph is created.

	"""
	if weights is not None:
		assert N == len(weights) + 1, (
			"Length of weights array is {}, not compatible with "
			"{} vertices.".format(len(weights), N))
	A = adj_matrix_path(N, weights=weights, directed=directed)
	coords = coords_path(N)
	return A, coords

def make_grid(rows, columns, weights_r=None, weights_c=None):
	"""Create a grid graph.

	Parameters
	----------
	rows: int
		Number of rows in the grid.
	columns: int
		Number of columns in the grid.
	weights_r: array, default=None
		Weights in the rows.
	weights_c: array, default=None
		Weights in the columns.

	"""
	A1, coords1 = make_path(columns, weights=weights_c)
	A2, coords2 = make_path(rows, weights=weights_r)

	N1 = len(A1)
	N2 = len(A2)

	# Using the property that the grid graph is the cartesian product
	# of two path graphs.
	A = np.kron(A1, np.eye(N2)) + np.kron(np.eye(N1), A2)
	coords = list()
	for c1 in coords1[:, 0].ravel():
		for c2 in coords2[:, 0].ravel():
			coords.append([c1, c2])
	coords = np.array(coords)

	return A, coords

def are(x, xr):
    """Average reconstruction error (ARE).

    Parameters
    ----------
	x : array
		Original graph signal.
	xr : array
		Reconstructed graph signal.
    """
    return np.linalg.norm(x - xr, ord=2) / np.linalg.norm(x, ord=2)

def reconstruct_from_p(x, U, Uinv, p):
	"""Reconstruct a graph signal from p frequency components.

	The `p` frequency components are largest in magnitude.

	Parameters
	----------
	x : 1D array
		Original graph signal.
	U : 2D array
		Eigenbasis of the chosen shift operator.
	Uinv : 2D array
		Inverse of matrix `U`.
	p : int
		Number of the largest (in magnitude) frequency components of the graph
		signal to use in the reconstruction.

	"""
	x_ = x.reshape((len(U), 1))
	xx_ = Uinv @ x_
	idx = np.argsort(np.abs(xx_).ravel())[::-1]
	out = U[:, idx[:p]] @ xx_[idx[:p]]
	return out

def are_vs_p_curve(image, A, verbose=True, start_p=10, step_p=50,
				   are_threshold=0.1):
	"""Compute the data for a `ARE vs p` curve for an image signal.

	The graph signal spectral decomposition is taken as in GSP_L, i.e.
	using the graph Laplacian as the shift operator.

	Also returns the smallest value of `p` for which the ARE is
	smaller than `are_threshold`.

	Parameters
	----------
	image : 2D array
		Pixel values from the input image. This is taken as a graph signal
		on a grid graph.
	A : 2D array
		Adjacency matrix of the underlying graph.
	start_p : int, default=10
		Initial value of `p` in the ARE computations.
	step_p : int, default=50
		Step used to increment the value of `p` in the ARE computations
		at each iteration.
	are_threshold : float, default=0.1
		Used to define the output value `smallest_p`, which is the smallest
		value of `p` for which the ARE is smaller than `are_threshold`.
	verbose : bool, default=True

	Return
	------
	p_values : array
	are_values : array
	smallest_p : int
		Smallest value of `p` for which the ARE is smaller
		than `are_threshold`.
	idx : boolean array
		Boolean array with the same length as `are_values`, holding True values
		wherever the ARE is smaller than `are_threshold`.

	"""
	im = copy.deepcopy(image)
	D = np.diag(np.sum(A, axis=1))
	L = D - A

	if verbose:
		print("Decomposing L.")
	_, U = np.linalg.eig(L)

	if verbose:
		print("Inverting U.")
	Uinv = np.linalg.inv(U)

	p_values = np.arange(start_p, len(U))[::step_p]
	are_values = list()

	for p in (tqdm.tqdm(p_values) if verbose else p_values):
		xr = reconstruct_from_p(im.ravel(), U, Uinv, p=p)
		N, M = im.shape
		imr = xr.reshape((N, M))
		are_values.append(are(im.ravel(), imr.ravel()))

	idx = np.array(are_values) < are_threshold
	smallest_p = p_values[idx][0]
	return p_values, are_values, smallest_p, idx
