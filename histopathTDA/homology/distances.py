'''
Distances
=========

This module contains functions that computing the distance between
persistence diagrams, which has:
- Wasserstein distance
- Bottleneck distance
- Optimal Transport distance
- Lp_norm distance
'''

import warnings
import numpy as np
from scipy import optimize   # type: ignore
import scipy.spatial.distance as sc   # type: ignore
import gudhi  # type: ignore
try:
    import ot  # type: ignore
except ImportError:
    print("POT (Python Optimal Transport) package is not installed. Try to run $ conda install -c conda-forge pot ; or $ pip install POT")


def wassertein(dgm1, dgm2, matching=False, dim=1):
    """
    Perform the Wasserstein distance matching between persistence diagrams for single dimentions.
    Assumes first two columns of dgm1 and dgm2 are the coordinates of the persistence
    points, but allows for other coordinate columns (which are ignored in
    diagonal matching).
    See the `distances` notebook for an example of how to use this.
    Parameters
    ------------
    dgm1:
        list of list containg for each birth/death pairs for PD 1
    dgm2:
        list of list containg for each birth/death paris for PD 2
    matching: bool, default False
        if True, return matching information and cross-similarity matrix
    dim: number
        PD dimension, default is 1
    Returns
    ---------
    d: float
        Wasserstein distance between dgm1 and dgm2
    (matching, D): Only returns if `matching=True`
        (tuples of matched indices, (N+M)x(N+M) cross-similarity matrix)
    """
    
    # convert dgm into specific dimension
    dgm1 = dgm1[dim][:, :2]
    dgm2 = dgm1[dim][:, :2]

    S = np.array(dgm1)
    M = min(S.shape[0], S.size)
    if S.size > 0:
        S = S[np.isfinite(S[:, 1]), :]
        if S.shape[0] < M:
            warnings.warn(
                "dgm1 has points with non-finite death times;" +
                "ignoring those points"
            )
            M = S.shape[0]
    T = np.array(dgm2)
    N = min(T.shape[0], T.size)
    if T.size > 0:
        T = T[np.isfinite(T[:, 1]), :]
        if T.shape[0] < N:
            warnings.warn(
                "dgm2 has points with non-finite death times;" +
                "ignoring those points"
            )
            N = T.shape[0]

    if M == 0:
        S = np.array([[0, 0]])
        M = 1
    if N == 0:
        T = np.array([[0, 0]])
        N = 1
    # Step 1: Compute CSM between S and dgm2, including points on diagonal
    # DUL = metrics.pairwise.pairwise_distances(S, T)
    # Step 1: Compute CSM between S and T, including points on diagonal
    # L Infinity distance
    Sb, Sd = S[:, 0], S[:, 1]
    Tb, Td = T[:, 0], T[:, 1]
    D1 = np.abs(Sb[:, None] - Tb[None, :])
    D2 = np.abs(Sd[:, None] - Td[None, :])
    DUL = np.maximum(D1, D2)

    # Put diagonal elements into the matrix
    # Rotate the diagrams to make it easy to find the straight line
    # distance to the diagonal
    cp = np.cos(np.pi / 4)
    sp = np.sin(np.pi / 4)
    R = np.array([[cp, -sp], [sp, cp]])
    S = S[:, 0:2].dot(R)
    T = T[:, 0:2].dot(R)
    D = np.zeros((M + N, M + N))
    D[0:M, 0:N] = DUL
    UR = np.max(D) * np.ones((M, M))
    np.fill_diagonal(UR, S[:, 1])
    D[0:M, N:(N + M)] = UR
    UL = np.max(D) * np.ones((N, N))
    np.fill_diagonal(UL, T[:, 1])
    D[M:(N + M), 0:N] = UL

    # Step 2: Run the hungarian algorithm
    matchi, matchj = optimize.linear_sum_assignment(D)
    matchdist = np.sum(D[matchi, matchj])

    if matching:
        matchidx = [(i, j) for i, j in zip(matchi, matchj)]
        return matchdist, (matchidx, D)

    return matchdist


def bottleneck(dgm1, dgm2, dim=1):
    """
    Perform the bottleneck distance matching between persistence diagrams for single dimentions.
    Assumes first two columns of dgm1 and dgm2 are the coordinates of the persistence
    points, but allows for other coordinate columns (which are ignored in
    diagonal matching).
    Points at infinity and on the diagonal are supported.
    ------------
    dgm1:
        (numpy array of shape (m,2))list of list containg for each birth/death pairs for PD 1
    dgm2:
        list of list containg for each birth/death paris for PD 2
    dim: number
        PD dimension, default is 1
    Returns
    ---------
    D: float
        bottleneck distance between dgm1 and dgm2
    """

    # convert dgm into specific dimension
    dgm1 = dgm1[dim][:, :2]
    dgm2 = dgm1[dim][:, :2]

    D = gudhi.bottleck_distance(dgm1, dgm2)
    return D


# helper function for OT distance computation
def _proj_on_diag(X):
    '''
    directly use the points on diagram, while in paper they use histograms
    :param X: (n x 2) array encoding the points of a persistent diagram.
    :returns: (n x 2) array encoding the (respective orthogonal) projections of the points onto the diagonal
    '''
    Z = (X[:, 0] + X[:, 1]) / 2.0
    return np.array([Z, Z]).T


def _build_dist_matrix(X, Y, order=2., internal_p=2.):
    '''
    :param X: (n x 2) numpy.array encoding the (points of the) first diagram.
    :param Y: (m x 2) numpy.array encoding the second diagram.
    :param internal_p: Ground metric (i.e. norm l_p).
    :param order: exponent for the Wasserstein metric.
    :returns: (n+1) x (m+1) np.array encoding the cost matrix C. 
                For 1 <= i <= n, 1 <= j <= m, C[i,j] encodes the distance between X[i] and Y[j], while C[i, m+1] (resp. C[n+1, j]) encodes the distance (to the p) between X[i] (resp Y[j]) and its orthogonal proj onto the diagonal.
                note also that C[n+1, m+1] = 0  (it costs nothing to move from the diagonal to the diagonal).
    '''
    Xdiag = _proj_on_diag(X)
    Ydiag = _proj_on_diag(Y)
    if np.isinf(internal_p):
        C = sc.cdist(X, Y, metric='chebyshev')**order
        Cxd = np.linalg.norm(X - Xdiag, ord=internal_p, axis=1)**order
        Cdy = np.linalg.norm(Y - Ydiag, ord=internal_p, axis=1)**order
    else:
        C = sc.cdist(X, Y, metric='minkowski', p=internal_p)**order
        Cxd = np.linalg.norm(X - Xdiag, ord=internal_p, axis=1)**order
        Cdy = np.linalg.norm(Y - Ydiag, ord=internal_p, axis=1)**order
    Cf = np.hstack((C, Cxd[:, None]))
    Cdy = np.append(Cdy, 0)

    Cf = np.vstack((Cf, Cdy[None, :]))

    return Cf


def _perstot(X, order, internal_p):
    '''
    :param X: (n x 2) numpy.array (points of a given diagram).
    :param internal_p: Ground metric on the (upper-half) plane (i.e. norm l_p in R^2); Default value is 2 (Euclidean norm).
    :param order: exponent for Wasserstein. Default value is 2.
    :returns: float, the total persistence of the diagram (that is, its distance to the empty diagram).
    '''
    Xdiag = _proj_on_diag(X)
    return (np.sum(np.linalg.norm(X - Xdiag, ord=internal_p, axis=1)**order))**(1.0 / order)


# helper function for OT distance computation
def ot_dist(dgm1, dgm2, matching=False, order=2.0, internal_p=2.0, dim=1):
    '''
    :param dgm1: (n x 2) numpy.array encoding the (finite points of the) first diagram. Must not contain essential points (i.e. with infinite coordinate).
    :param dgm2: (m x 2) numpy.array encoding the second diagram.
    :param internal_p: Ground metric on the (upper-half) plane (i.e. norm l_p in R^2); Default value is 2 (euclidean norm).
    :param order: exponent for Wasserstein; Default value is 2.
    :returns: the Wasserstein distance of order q (1 <= q < infinity) between persistence diagrams with respect to the internal_p-norm as ground metric.
    :rtype: float
    '''

    # convert dgm into specific dimension
    dgm1 = dgm1[dim][:, :2]
    dgm2 = dgm1[dim][:, :2]

    n = len(dgm1)
    m = len(dgm2)

    # handle empty diagrams
    if dgm1.size == 0:
        if dgm2.size == 0:
            return 0.
        else:
            return _perstot(dgm2, order, internal_p)
    elif dgm2.size == 0:
        return _perstot(dgm1, order, internal_p)

    M = _build_dist_matrix(dgm1, dgm2, order=order, internal_p=internal_p)
    a = np.full(n + 1, 1.0 / (n + m))  # weight vector of the input diagram. Uniform here.
    a[-1] = a[-1] * m                # normalized so that we have a probability measure, required by POT
    b = np.full(m + 1, 1.0 / (n + m))  # weight vector of the input diagram. Uniform here.
    b[-1] = b[-1] * n                # so that we have a probability measure, required by POT

    if matching:
        P = ot.emd(a=a, b=b, M=M, numItermax=2000000)
        ot_cost = np.sum(np.multiply(P, M))
        P[-1, -1] = 0  # Remove matching corresponding to the diagonal
        match = np.argwhere(P)
        # Now we turn to -1 points encoding the diagonal
        match[:, 0][match[:, 0] >= n] = -1
        match[:, 1][match[:, 1] >= m] = -1
        return ot_cost ** (1.0 / order), match

    # Comptuation of the otcost using the ot.emd2 library.
    # Note: it is the Wasserstein distance to the power q.
    # The default numItermax=100000 is not sufficient for some examples with 5000 points, what is a good value?
    ot_cost = (n + m) * ot.emd2(a, b, M, numItermax=2000000)

    return ot_cost ** (1.0 / order)


def l_norm(dgm1, dgm2, norm=2):
    """
    Calculate euclidian distance between two persistance diagrams.
    Parameters
    ___________
    dgm1:
        ndarray of persistence diagram 1
    dgm2:
        ndarray of persistence diagram 2
    norm:
        l-norm of distance calculation, can be 1 or 2
    Returns
    ________
    d: float
        euclidian distance between dgm1 and dgm2
    """
    # Vectorize persistance diagrams into one dimensional ndarrays
    v_dgm1 = dgm1.flatten()
    v_dgm2 = dgm2.flatten()
    # check norm value
    if norm != 1 and norm != 2:
        print("Invalid norm value")
        return
    # calculate distance
    n = v_dgm1.size
    s = 0.0
    d = sum(((np.abs(v_dgm2[i] - v_dgm1[i]))**norm)for i in range(0, n))
    d = d**(1.0 / norm)
    return d
