import numpy as np
from itertools import combinations


def index_covij(n):
    total = n * (n + 1) // 2
    m = np.triu(np.random.randn(n) > 0).astype(int)

    oldind = np.argwhere(m)
    newind = np.argwhere(np.argwhere(m))

    newlist = np.delete(oldind, 0, axis=1)

    ind_i, ind_j = np.argwhere(m).T
    ind_covij = [None] * total

    k = 0
    while k < len(ind_i):
        ind_covij[k] = np.zeros(ind_i[k])
        k = k + 1

    for l in range(len(ind_i)):
        i = ind_i[l]
        j = ind_j[l]
        r = np.where(m[:, i] & m[:, j])[0]
        length = len(r)

        if i == 0:
            R = np.column_stack((i, np.tile(r, (1, length)))).T
        else:
            R = np.tile(r, (1, 2)).T

        if i == j:
            if i == 0:
                c = np.tile(i, (1, length))
                C = np.column_stack((i, c)).T
            else:
                c = np.tile(i, (1, length))
                C = np.tile(c, (1, 2))
        else:
            if i == 0:
                vec = np.array([i, j])
                C = np.tile(vec, (1, length)).T
            else:
                vec = np.array([i, j])
                C = np.tile(vec, (1, 2))

        ind = np.ravel_multi_index((R, C), dims=m.shape)

        if i != 0:
            ind = ind.reshape(length, 2)

        index_r = np.where(np.isin(ind, newlist[:, 0]))

        if index_r:
            pos = [np.where(newlist == x)[0][0] for x in ind[index_r]]
            ind[index_r] = newlist[pos, 1]

        ind_covij[l] = ind

    return ind_covij


def indices_convertback(n):
    total = n * (n + 1) // 2
    m = np.triu(np.ones((n, n), dtype=bool))
    A = np.zeros((n, n))
    A[m] = np.arange(1, total + 1)
    A = A.T
    diag_A = np.diag(A)
    m1 = np.tril(np.ones_like(A), k=-1)
    index = np.concatenate((diag_A, A[m1])).tolist()
    return index

def indices_convert(n):
    m = np.triu(np.ones((n, n), dtype=bool))
    total = n * (n + 1) // 2

    index_diag = np.ravel_multi_index((np.arange(n), np.arange(n)), dims=m.shape)
    values = np.arange(1, total + 1)
    values_var = values[:n]
    values = values[n:]

    index_cov = []
    values_cov = []

    for i in range(1, n):
        index_cov_i = np.ravel_multi_index((np.tile(i, n - i), np.arange(i + 1, n + 1)), dims=m.shape)
        values_cov_i = values[:len(index_cov_i)]
        values = values[len(index_cov_i):]
        m.flat[index_cov_i] = values_cov_i

    m.flat[index_diag] = values_var
    ind = np.triu(np.ones((n, n), dtype=bool))
    index = m[ind].tolist()
    
    return index

def indcov_wijkl(n_x):
    n_w2 = n_x * (n_x + 1) // 2
    cov_wijkl = [np.zeros((1, s - 1)) for s in range(n_w2, 0, -1)]
    cov_prior_wijkl = [np.zeros((1, s - 1)) for s in range(n_w2, 0, -1)]
    ind_wijkl = [np.zeros((s - 1, 4), dtype=int) for s in range(n_w2, 0, -1)]
    ind_cov = [np.zeros((s - 1, 4), dtype=int) for s in range(n_w2, 0, -1)]
    ind_cov_prior = [np.zeros((s - 1, 4), dtype=int) for s in range(n_w2, 0, -1)]

    v = np.arange(1, n_x + 1)
    n_w = n_x
    I1 = np.vstack((np.column_stack((np.arange(1, n_w + 1), np.arange(1, n_w + 1))), np.array(list(combinations(v, 2)))))
    I2 = I1.copy()
    I1 = I1[:-1, :]

    for i in range(n_w2 - 1):
        ind_wijkl[i][:, 0:2] = np.tile(I1[i, :], (ind_wijkl[i].shape[0], 1))

    i = 0
    j = 1
    while i <= n_w2 - 1:
        ind_wijkl[i][:, 2:4] = I2[j:, :]
        j = j + 1
        i = i + 1

    return ind_wijkl

def ind_mean(n_x, ind_wijkl):
    n_w2 = n_x * (n_x + 1) // 2
    ind_mu = [ind_wijkl[i][:, [0, 3, 1, 2]] for i in range(n_w2 - 1)]
    return ind_mu

def ind_covariance(n_w2, n_w, ind_wijkl):
    ind_cov = []

    for i in range(n_w2 - 1):
        m_ijkl = ind_wijkl[i]
        r = m_ijkl.shape[0]
        ind_cov_i = np.zeros((r, 4), dtype=int)

        for j in range(r):
            ijkl = m_ijkl[j]
            ind_cov_i[j, 0] = n_w * (ijkl[2] - 1) + ijkl[0]
            ind_cov_i[j, 1] = n_w * (ijkl[3] - 1) + ijkl[1]
            ind_cov_i[j, 2] = n_w * (ijkl[2] - 1) + ijkl[1]
            ind_cov_i[j, 3] = n_w * (ijkl[3] - 1) + ijkl[0]

        ind_cov.append(ind_cov_i)

    return ind_cov



def indcov_priorWp(ind_wijkl, n_w2, n_x):
    ind_cov_prior = []

    for i in range(n_w2 - 1):
        m_ijkl = ind_wijkl[i]
        r = m_ijkl.shape[0]
        v = np.arange(1, n_x + 1)
        ind_cov_prior_i = np.zeros((r, 4), dtype=int)

        for j in range(r):
            ijkl = m_ijkl[j]
            if ijkl[0] == ijkl[2]:
                ind_cov_prior_i[j, 0] = np.where(v == ijkl[0])[0][0]
            else:
                diff = abs(ijkl[2] - ijkl[0])
                ind_cov_prior_i[j, 0] = diff + n_x

            if ijkl[1] == ijkl[3]:
                ind_cov_prior_i[j, 1] = np.where(v == ijkl[1])[0][0]
            else:
                diff = abs(ijkl[3] - ijkl[1])
                ind_cov_prior_i[j, 1] = diff + n_x

            if ijkl[0] == ijkl[3]:
                ind_cov_prior_i[j, 2] = np.where(v == ijkl[0])[0][0]
            else:
                diff = abs(ijkl[3] - ijkl[0])
                ind_cov_prior_i[j, 2] = diff + n_x

            if ijkl[1] == ijkl[2]:
                ind_cov_prior_i[j, 3] = np.where(v == ijkl[1])[0][0]
            else:
                diff = abs(ijkl[2] - ijkl[1])
                ind_cov_prior_i[j, 3] = diff + n_x

        ind_cov_prior.append(ind_cov_prior_i)

    return ind_cov_prior





