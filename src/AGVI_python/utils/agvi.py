import numpy as np
from itertools import combinations
from gma import cov1234  

def priorcov_wijkl(m_wsqhat, ind_cov_prior, n_w2):
    cov_prior_wijkl = [None] * (n_w2 - 1)
    i = 0
    s = n_w2

    while i < (n_w2 - 1):
        cov_prior_wijkl[i] = [0] * (s - 1)
        i = i + 1
        s = s - 1

    i = 0

    while i < (n_w2 - 1):
        ind_Mpr = ind_cov_prior[i]
        j = 0

        while j < len(ind_Mpr):
            cov_prior_wijkl[i][j] = (
                m_wsqhat[ind_Mpr[j][0]] * m_wsqhat[ind_Mpr[j][1]]
                + m_wsqhat[ind_Mpr[j][2]] * m_wsqhat[ind_Mpr[j][3]]
            )
            j = j + 1

        n1 = (n_w2 - 1) - len(cov_prior_wijkl[i])

        if n1 > 0:
            add_zeros = [0] * n1
        else:
            add_zeros = []

        cov_prior_wijkl[i] = add_zeros + cov_prior_wijkl[i]
        i = i + 1

    return cov_prior_wijkl

def covwp(P_P, m_wsqhat, n_x, n_w2, n_w):
    v = np.arange(1, n_x + 1)
    s_wsqhat = P_P
    PX_wp = np.zeros(s_wsqhat.shape)

    ind12 = list(combinations(v, 2))
    i = 0
    j = 0

    while i < n_w2:
        if i < n_w:
            PX_wp[i, i] = 2 * m_wsqhat[i] ** 2 + 3 * s_wsqhat[i, i]
        else:
            PX_wp[i, i] = s_wsqhat[i, i] + (m_wsqhat[i] ** 2 / (m_wsqhat[ind12[j][0] - 1] * m_wsqhat[ind12[j][1] - 1] + m_wsqhat[i] ** 2)) * s_wsqhat[i, i] + m_wsqhat[ind12[j][0] - 1] * m_wsqhat[ind12[j][1] - 1] + m_wsqhat[i] ** 2
            j = j + 1
        i = i + 1

    return PX_wp

def var_wpy(cov_wijkl, s_wii_y, s_wiwj_y):
    cov_wpy = np.concatenate(cov_wijkl, axis=1)
    PX_wpy = np.diag(np.concatenate([s_wii_y, s_wiwj_y]))
    s_wpy = np.zeros(PX_wpy.shape)
    s_wpy[:-1, 1:] = cov_wpy
    PX_wpy += s_wpy
    PX_wpy = np.triu(PX_wpy) + np.triu(PX_wpy, 1)
    return PX_wpy


def cov_wijkl(PX_wy, EX_wy, n_w2, ind_cov, ind_mu):
    cov_wijkl = [None] * (n_w2 - 1)
    i = 0
    s = n_w2

    while i < (n_w2 - 1):
        cov_wijkl[i] = [0] * (s - 1)
        i = i + 1
        s = s - 1

    i = 0

    while i < (n_w2 - 1):
        ind_C = ind_cov[i]
        ind_M = ind_mu[i]
        j = 0

        while j < len(ind_C):
            cov_wijkl[i][j] = cov1234(ind_C[j], ind_M[j], PX_wy, EX_wy)
            j = j + 1

        n1 = (n_w2 - 1) - len(cov_wijkl[i])

        if n1 > 0:
            add_zeros = [0] * n1
        else:
            add_zeros = []

        cov_wijkl[i] = add_zeros + cov_wijkl[i]
        i = i + 1

    return cov_wijkl


def meanvar_w2pos(EX_wy, PX_wy, cwiwjy, P_wy, n_w2hat, n_x):
    m_wii_y = EX_wy**2 + np.diag(PX_wy)
    s_wii_y = 2 * np.diag(PX_wy)**2 + 4 * np.diag(PX_wy) * EX_wy**2

    m_wiwj_y = np.zeros(n_w2hat - n_x)
    s_wiwj_y = np.zeros(n_w2hat - n_x)

    i, j, k = 0, 0, 0

    while i < n_x - 1:
        m_wiwj_y[k] = EX_wy[i] * EX_wy[j + 1] + cwiwjy[k]
        s_wiwj_y[k] = P_wy[i] * P_wy[j + 1] + cwiwjy[k]**2 + 2 * cwiwjy[k] * EX_wy[i] * EX_wy[j + 1] + P_wy[i] * EX_wy[j + 1]**2 + P_wy[j + 1] * EX_wy[i]**2
        j = j + 1
        k = k + 1
        if j == n_x:
            i = i + 1
            j = i

    return m_wii_y, s_wii_y, m_wiwj_y, s_wiwj_y


def PtoL(Cov_L_P, E_P, P_P, EL_pr, PL_pr, E_Pw_y, P_Pw_y):
    Jc = Cov_L_P / (P_P + 1e-08)
    
    EL_pos = EL_pr + np.dot(Jc, (E_Pw_y - E_P))
    PL_pos = PL_pr + np.dot(np.dot(Jc, (P_Pw_y - P_P)), Jc.T)
    
    return EL_pos, PL_pos


def agviSmoother(E_wp, s_wsqhat, PX_wp, EX_wpy, PX_wpy, E_P, P_P):
    E_Wp_prior = E_wp
    C_Wp_W2hat = s_wsqhat
    P_Wp_prior = PX_wp
    E_Wp_pos = EX_wpy
    P_Wp_pos = PX_wpy

    J = C_Wp_W2hat / (P_Wp_prior + 1e-08)

    ES = E_P + np.dot(J, (E_Wp_pos - E_Wp_prior))
    PS = P_P + np.dot(np.dot(J, (P_Wp_pos - P_Wp_prior)), J.T)

    return ES, PS