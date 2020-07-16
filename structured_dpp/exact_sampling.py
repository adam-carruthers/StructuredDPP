"""
Copied code from DPPy https://github.com/guilgautier/DPPy
Full credit to them, their library is very well written.
I just wanted to reduce dependencies of my project.
"""
import numpy as np


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    .. seealso::
        `Scikit learn source code <https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/utils/validation.py#L763>`_
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def dpp_eigvals_selector(eigvals, random_state=None):
    """ Subsample eigenvalues V of the 'L' kernel """

    rng = check_random_state(random_state)

    selected_indices = []
    for i, eigval in enumerate(eigvals):
        if rng.rand() < eigval/(eigval+1):
            selected_indices.append(i)

    return selected_indices


def k_dpp_eigvals_selector(eigvals, k, E_poly=None, random_state=None):
    """ Subsample eigenvalues V of the 'L' kernel to build a projection DPP with kernel V V.T from which sampling is easy. The selection is made based a realization of Bernoulli variables with parameters the eigenvalues of 'L' and evalutations of the elementary symmetric polynomials.
    :param eigvals:
        Collection of eigen values of 'L' (likelihood) kernel.
    :type eigvals:
        list, array_like
    :param k:
        Size :math:`k` of :math:`k`-DPP
    :type size:
        int
    :param E_poly:
        Evaluation of symmetric polynomials in the eigenvalues
    :type E_poly:
        array_like
    :return:
        Selected eigenvalue indices
    :rtype:
        array_like
    .. seealso::
        - :cite:`KuTa12` Algorithm 8
        - :func:`elementary_symmetric_polynomials <elementary_symmetric_polynomials>`
    """

    rng = check_random_state(random_state)

    # as in np.linalg.matrix_rank
    # except np.linalg.matrix_rank uses N, but for a SDPP this is too big sooooo we'll just use the number of features
    tol = np.max(eigvals) * eigvals.size * np.finfo(np.float).eps
    rank = np.count_nonzero(eigvals > tol)
    if k > rank:
        raise ValueError('size k={} > rank={}'.format(k, rank))

    if E_poly is None:
        E_poly = elementary_symmetric_polynomials(eigvals, k)

    ind_selected = np.zeros(k, dtype=int)
    for n in range(eigvals.size, 0, -1):

        if rng.rand() < eigvals[n - 1] * E_poly[k - 1, n - 1] / E_poly[k, n]:
            k -= 1
            ind_selected[k] = n - 1
            if k == 0:
                break

    return np.arange(eigvals.size)[ind_selected]


# Evaluate the elementary symmetric polynomials
def elementary_symmetric_polynomials(eigvals, size):
    """ Evaluate the elementary symmetric polynomials :math:`e_k` in the eigenvalues :math:`(\\lambda_1, \\cdots, \\lambda_N)`.
    :param eigvals:
        Collection of eigenvalues :math:`(\\lambda_1, \\cdots, \\lambda_N)` of the similarity kernel :math:`L`.
    :type eigvals:
        list
    :param size:
        Maximum degree of elementary symmetric polynomial.
    :type size:
        int
    :return:
        :math:`[E_{kn}]_{k=0, n=0}^{\text{size}, N}`
        :math:`E_{kn} = e_k(\\lambda_1, \\cdots, \\lambda_n)`
    :rtype:
        array_like
    .. seealso::
        - :cite:`KuTa12` Algorithm 7
        - `Wikipedia <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_
    """

    # Initialize output array
    N = eigvals.size
    E_poly = np.zeros((size + 1, N + 1))
    E_poly[0, :] = 1.0

    # Recursive evaluation
    for l in range(1, size + 1):
        for n in range(1, N + 1):
            E_poly[l, n] = E_poly[l, n-1] + eigvals[n - 1] * E_poly[l - 1, n - 1]

    return E_poly