import errno
import os
import numpy as np


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_non_dominated(candidates):
    """
    This function returns the non-dominated subset of elements.
    :param candidates: The input set of candidate vectors.
    :return: The non-dominated subset of this input set.
    """
    candidates = np.array(list(candidates))
    #print(candidates)
    # sort candidates by decreasing sum of coordinates
    candidates = candidates[candidates.sum(1).argsort()[::-1]]
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    nd = np.ones(candidates.shape[0], dtype=bool)
    for i in range(candidates.shape[0]):
        # process each point in turn
        n = candidates.shape[0]
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        nd[i+1:n] = (candidates[i+1:] > candidates[i]).any(1)
        # keep points non-dominated so far
        candidates = candidates[nd[:n]]
    #print('non-dom', candidates)
    return candidates

def is_dominated(candidate, vectors):
    """
    This function checks if a candidate vector is dominated by another vector in the set.
    :param candidate: The candidate vector.
    :param vectors: A set of vectors with type Tuple.
    :return: Boolean whether the candidate is dominated.
    """
    if not vectors:
        return False
    else:
        for vector in vectors:
            vector = np.array(vector)
            if not (candidate == vector).all():
                dominated = candidate <= vector
                if dominated.all():
                    return True
        return False
