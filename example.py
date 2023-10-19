import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(1)

def data_generate(T, n_x, sW, sV):
    # Check if sW and sV are non-negative
    assert sW >= 0, "sW must be non-negative."
    assert sV >= 0, "sV must be non-negative."

    YT = np.zeros((n_x, T))
    x_true = np.zeros((n_x, T))
    x_true[:, 0] = 0
    w = np.dot(sW, np.random.randn(n_x, T))
    v = np.dot(sV, np.random.randn(n_x, T))
    for t in range(1, T):
        A_t = 0.8 - 0.1 * np.sin(7 * np.pi * t / T)  # Transition equation
        C_t = 1 - 0.99 * np.sin(100 * np.pi * t / T)  # Observation equation
        x_true[:, t] = A_t * x_true[:, t - 1] + w[:, t]
        YT[:, t] = C_t * x_true[:, t] + v[:, t]
    
    return YT, x_true
    
def initialization(T, n_x, n_w2hat, mw2, sw2):
    EX = np.zeros((3, T))  # X = [X  W  \overline{W^2}]
    EX[:, 0] = np.transpose([0, np.nan, mw2])  # Initial mean is zero
    PX = np.zeros((3, 3, T))
    PX[:, :, 0] = np.diag([100, np.nan, sw2 ** 2])  # Initial variance is 100
    return EX, PX

def KFpredict(A, EX, PX):
    # Prediction step
    Ep = np.transpose([A * EX[0], 0])  # Mean of W is zero
    Ep = Ep.reshape(Ep.shape[0], 1)
    m_w2hat = EX[2]
    s_w_sq = m_w2hat

    Sp = np.array([[A ** 2 * PX[0, 0] + s_w_sq, s_w_sq],
                   [s_w_sq, s_w_sq]])
    return Ep, Sp

def KFupdate(C, R, YT, Ep, Sp):
    # Update step
    SY = C @ Sp @ C.T + R
    SYX = Sp @ C.T
    my = C @ Ep
    K = SYX / SY
    K = K.reshape(K.shape[0], 1)
    e = YT - my
    e = e.reshape(e.shape[0], 1)
    SYX = SYX.reshape(SYX.shape[0], 1)
    EX1 = Ep + K @ e
    PX1 = Sp - K @ SYX.T
    EX1 = np.reshape(EX1, (EX1.shape[0],))
    return EX1, PX1

def KFupdate2(t, EX, PX):
    s_w2_sq = 2 * EX[2, t - 1] ** 2
    m_w2 = EX[1, t] ** 2 + PX[1, 1, t]
    s_w2 = 2 * PX[1, 1, t] ** 2 + 4 * PX[1, 1, t] * EX[1, t] ** 2
    my1 = EX[2, t - 1]
    SYX1 = PX[2, 2, t - 1]
    return s_w2_sq, m_w2, s_w2, my1, SYX1

def KSmoother(s_w2_sq, m_w2, s_w2, my1, SYX1, EX, PX, t):
    E_W2_pos = m_w2
    E_W2_prior = my1
    C_W2_W2hat = SYX1
    P_W2_prior = 3 * PX[2, 2, t - 1] + s_w2_sq
    P_W2_pos = s_w2
    J = C_W2_W2hat / P_W2_prior
    EX[2, t] = EX[2, t - 1] + J * (E_W2_pos - E_W2_prior)
    PX[2, 2, t] = PX[2, 2, t - 1] + J ** 2 * P_W2_pos - C_W2_W2hat ** 2 / P_W2_prior
    return EX, PX

def hypothesis_test(estim_sigma_w, estim_sigma_w_P, sW, alpha=0.05):
    """
    Perform a hypothesis test to determine whether estim_sigma_w is significantly different from sW^2.

    Parameters:
    - estim_sigma_w (float): Estimated value of sigma_w.
    - estim_sigma_w_P (float): Variance of the estimate of sigma_w.
    - sW (float): True value of sW.
    - alpha (float, optional): Significance level (default is 0.05).

    Returns:
    - result (str): The result of the hypothesis test.
    """
    df = 1  # Degrees of freedom for the test (fixed for this case)
    t_critical = stats.t.ppf(1 - alpha / 2, df)  # t-critical value for a two-tailed test

    # Calculating the test statistic
    x_bar = estim_sigma_w
    mu = sW ** 2
    s = (estim_sigma_w_P)**0.5
    t_statistic = (x_bar - mu) / (s / (1**0.5))

    # Testing the null hypothesis
    if abs(t_statistic) > t_critical:
        return f"Reject the null hypothesis at the {100 * (1 - alpha):.2f}% significance level."
    else:
        return f"Fail to reject the null hypothesis at the {100 * (1 - alpha):.2f}% significance level."


def plot_sigma_w(t, sW, EX, PX):
    """
    Plot the estimates of sigma_w over time.

    Parameters:
    - t (numpy array): Time steps.
    - sW (float): True value of sigma_w.
    - EX (numpy array): State estimates.
    - PX (numpy array): Covariance estimates.

    Returns:
    - None
    """
    t = np.arange(1, len(EX[2]))
    xw = EX[2, t]
    sX = np.sqrt(PX[2, 2, t])
    plt.figure()
    plt.plot(t, sW**2 * np.ones_like(t), '-.r', linewidth=1.5)
    plt.fill_between(t, xw + sX, xw - sX, color='g', alpha=0.2)
    plt.plot(t, xw, 'k')
    plt.legend(['True', '$\mu \pm \sigma$', '$\mu$'], loc='best')
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$\sigma^2_W$', fontsize=14)
    plt.show()

## Parameters
T = 1000   # Time-series length
n_x = 1    # Number of time series
n_w = n_x  # Number of process error terms

n_w2hat = n_x * (n_x + 1) / 2  # Total number of variance and covariance terms

# Q matrix
user_variance = 0.42 # 0.42, 1.35, 18.75 - Possible values for the true error variances
assert user_variance > 0, "Variance must be positive."

sW = np.sqrt(user_variance)  
Q = sW ** 2

# R matrix
QR_ratio = 10  # Q/R = (sigma_W)^2 / (sigma_V)^2
R = Q / QR_ratio
sV = np.sqrt(R)

## Data
YT, x_true = data_generate(T, n_x, sW, sV)

# plt.figure()
# plt.plot(YT[0, :], 'k')
# plt.xlabel('$t$', fontsize=14)
# plt.ylabel('$Y_t$', fontsize=14)
# plt.show()

## Initialization

mw2 = 1  # Initial mean for \overline{W^2}
sw2 = 0.5  # Initial variance for \overline{W^2}
assert mw2 > 0, "Initial mean must be positive."
assert sw2 > 0, "Initial variance must be positive."

EX, PX = initialization(T, n_x, n_w2hat, mw2, sw2)
# print(EX[:, 0])
# print(PX[:, :, 0])
## State Estimation

for t in range(1, T):
    A = 0.8 - 0.1 * np.sin(7 * np.pi * t / T)  # Transition equation
    Ep, Sp = KFpredict(A, EX[:, t - 1], PX[:, :, t - 1])

    C = np.array([1 - 0.99 * np.sin(100 * np.pi * t / T), 0])  # Observation equation
    EX1, PX1 = KFupdate(C, R, YT[:, t], Ep, Sp)
    
    EX[0:2, t] = EX1
    PX[0:2, 0:2, t] = PX1
    
    # 2nd Update step
    s_w2_sq, m_w2, s_w2, my1, SYX1 = KFupdate2(t, EX, PX)

    # Smoother Step to update \overline{W^2}
    EX, PX = KSmoother(s_w2_sq, m_w2, s_w2, my1, SYX1, EX, PX, t)

    
## Getting the estimates
estim_sigma_w = EX[2, -1]
estim_sigma_w_P = PX[2, 2, -1]

## Hypothesis Testing

result = hypothesis_test(estim_sigma_w, estim_sigma_w_P, sW)
print(result)

## Plotting Sigma_W
plot_sigma_w(t, sW, EX, PX)


