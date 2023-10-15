import numpy as np

def kf_predict(A, C, Q, R, y, P, Ep):
    # Predict step
    Sx = A @ P @ A.T
    Sx = (Sx + Sx.T) / 2
    Sx = Sx + Q
    Sp = np.block([[Sx, Q], [Q, Q]])
    Sp = (Sp + Sp.T) / 2

    # Observation matrix
    # C = np.block([[np.eye(n_x), np.zeros((n_x, n_x))]])
    
    SY = C @ Sp @ C.T + R
    SYX = Sp @ C.T
    my = C @ Ep
    K = SYX @ np.linalg.inv(SY + 1e-08)
    e = y - my  # YT[:, t]
    NIS = e.T @ np.linalg.pinv(SY) @ e

    # First Update step
    if np.isnan(y).any():
        EX_pos = Ep
        PX_pos = Sp
    else:
        EX_pos = Ep + K @ e
        PX_pos = Sp - K @ SYX.T

    return EX_pos, PX_pos, NIS
