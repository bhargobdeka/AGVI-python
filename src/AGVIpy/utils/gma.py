"""
Equations 4 -7 from the paper "The Gaussian multiplicative approximation for state-space models <https://onlinelibrary.wiley.com/doi/full/10.1002/stc.2904?casa_token=rXH8MRQlV1cAAAAA%3AulafSkWVNfg5kINZsVId-dRvNm382I0TZZe_qbv144e4Sq5qGbQhtTWdQCb32NhoD5UGmRZog6DExFU>"
"""

def xy(mx, my, Sx, Sy, Cxy):
    mxy = mx * my + Cxy
    Sxy = Sx * Sy + Cxy ** 2 + 2 * Cxy * mx * my + mx ** 2 * Sy + my ** 2 * Sx
    return mxy, Sxy

def Cxyz(my, mz, Cxy, Cxz):
    C_x_yz = Cxy * mz + Cxz * my
    return C_x_yz

def cov1234(ind_C, ind_M, PX_wy, EX_wy):
    V = (
        2 * PX_wy[ind_C[0]] * PX_wy[ind_C[1]] +
        2 * PX_wy[ind_C[2]] * EX_wy[ind_M[0]] * EX_wy[ind_M[1]] +
        2 * PX_wy[ind_C[3]] * EX_wy[ind_M[2]] * EX_wy[ind_M[3]]
    )
    return V
