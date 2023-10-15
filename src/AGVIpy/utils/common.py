import numpy as np

def sparse_approx(C, threshold):
    # Compute absolute values of entries in C
    abs_C = np.abs(C)

    # Apply threshold to absolute values
    abs_C[abs_C < threshold] = 0

    # Create a new matrix C_sp with the same shape as C
    C_sp = np.zeros_like(C)

    # Set diagonal entries of C_sp to the diagonal entries of C
    np.fill_diagonal(C_sp, np.diag(C))

    # Set off-diagonal entries of C_sp to the thresholded absolute values
    C_sp[~np.eye(C.shape[0], dtype=bool)] = abs_C[~np.eye(C.shape[0], dtype=bool)]

    return C_sp


def low_rank_kron(cov_matrix, explained_variance=0.99):
    # Compute the eigenvalue decomposition of the covariance matrix
    V, D = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues in descending order
    idx = np.argsort(-V)
    V = V[idx]
    D = D[:, idx][idx]

    # Compute the cumulative explained variance
    cumulative_variance = np.cumsum(V) / np.sum(V)

    # Determine the smallest rank that retains the desired percentage of variance
    r = np.where(cumulative_variance >= explained_variance)[0][0]

    # Compute the low-rank approximation
    low_rank_matrix = V[:r] @ np.diag(D[:r]) @ V[:r].T

    # Kronecker factorization
    U, D, V = np.linalg.svd(low_rank_matrix)
    k = np.linalg.matrix_rank(U)
    Ur = U[:, :r]
    Dr = np.diag(D[:r])
    Vr = V[:, :r]
    A = Ur @ np.sqrt(Dr)
    B = np.sqrt(Dr) @ Vr.T

    # Compute the Kronecker product
    kron_matrix = np.kron(A, B)

    return kron_matrix

def low_rank_approximation(Sigma, explained_variance):
    # Compute the eigenvalue decomposition of the covariance matrix
    V, D = np.linalg.eig(Sigma)

    # Sort the eigenvalues in descending order
    idx = np.argsort(-np.diag(D))
    V = V[:, idx]
    D = D[:, idx][idx]

    # Compute the cumulative explained variance
    cumulative_variance = np.cumsum(np.diag(D)) / np.sum(np.diag(D))

    # Determine the smallest rank that retains the desired percentage of variance
    rank = np.where(cumulative_variance >= explained_variance)[0][0]

    # Compute the low-rank approximation
    Sigma_low_rank = V[:, :rank] @ np.diag(D[:rank]) @ V[:, :rank].T

    return Sigma_low_rank

def make_PSD(Q_W, smallest_eigenvalue=1e-06):
    V, E = np.linalg.eig(Q_W)
    E = np.diag(E)
    v = E < 0
    m = np.sum(v)
    if m > 0:
        S = np.sum(v * E)
        W = (S ** 2 * 100) + 1
        P = 0.01  # User-defined smallest positive eigenvalue
        i = np.where(E < 0)
        C1 = E[i]
        E[i] = (P * (S - C1) * (S - C1) / W) + smallest_eigenvalue
    Q = V @ np.diag(E) @ V.T
    return Q

def make_PSDV2(B, min_eigenvalue=0.1):
    Q, d = np.linalg.eig(B)
    X_F = Q @ (np.maximum(d, min_eigenvalue) * Q.T)
    X_F = (X_F + X_F.T) / 2
    return X_F

