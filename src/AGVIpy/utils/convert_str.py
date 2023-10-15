from indices import indices_convert, indices_convertback
from gma import xy, Cxyz
import numpy as np

def convertstructure(mL, SL, n):
    ind = indices_convert(n)
    mL_new = mL[ind]
    SL_new = SL[ind]
    SL_new = SL_new[:, ind]
    return mL_new, SL_new

def convertback(E_P, P_P, Cov_L_P, n):
    ind = indices_convertback(n)
    E_P_old = E_P[ind]
    P_P_old = P_P[ind]
    P_P_old = P_P_old[:, ind]
    Cov_L_P_old = None
    if Cov_L_P is not None:
        Cov_L_P_old = Cov_L_P[ind]
        Cov_L_P_old = Cov_L_P_old[:, ind]
    return E_P_old, P_P_old, Cov_L_P_old




def LtoP(n, mL_mat, SL_mat, SL, ind_covij):
    total = n * (n + 1) // 2
    m = np.triu(np.ones(n, dtype=bool))
    E_P = np.zeros(total)
    P_P = np.zeros((total, total))
    Cov_L_P = np.zeros((total, total))

    M = mL_mat[m]
    ind_i, ind_j = np.where(mL_mat)
    for l in range(len(ind_i)):
        covxy = np.zeros(n)
        i = ind_i[l]
        j = ind_j[l]
        ind = ind_covij[l]
        for idx in range(len(ind)):
            ind_idx = ind[idx]
            covxy[idx] = np.diag(SL[ind_idx[0], ind_idx[1]])

        mv2_, Sv2_ = xy(mL_mat[:, i], mL_mat[:, j], SL_mat[:, i], SL_mat[:, j], covxy)
        E_P[l] = np.sum(mv2_)
        P_P[l, l] = np.sum(Sv2_)

        C = np.zeros((total, len(ind)))
        for k in range(len(ind)):
            C[:, k] = Cxyz(M[ind[k][0]], M[ind[k][1]], SL[:, ind[k][0]], SL[:, ind[k][1], :])

        Cov_L_P[:, l] = np.sum(C, axis=1)

    return E_P, P_P, Cov_L_P


