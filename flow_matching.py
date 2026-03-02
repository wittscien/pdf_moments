import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt


# Set precision if needed
mp.mp.dps = 50


def harmonic_number(n):
    return mp.nsum(lambda k: 1/mp.mpf(k), [1, n])


def gamma_n(n, CF=4/3):
    """
    One-loop anomalous dimension for non-singlet twist-2 operator.
    γ_n = 8 C_F [ H_n - 1/(2n(n+1)) - 3/4 ]
    """
    Hn = harmonic_number(n)
    return 8 * CF * (Hn - 1/(2*n*(n+1)) - mp.mpf(3)/4)


def B_n(n):
    """
    Finite part B_n as given in your formula.
    """
    Hn = harmonic_number(n)

    # first rational term
    term1 = (2 - 4*n**2*(n + 2)) / (n*(n + 1)**2)

    # harmonic number term
    term2 = 2*(2*n + 1)/(n*(n + 1)) * Hn

    # logs
    term3 = -4/n * mp.log(2)
    term4 = -3 * mp.log(3)

    # hypergeometric sum
    def summand(j):
        return (1/(j*(j-1)*2**j)) * mp.hyper([1/2, 1], [j], 1)

    sum_term = mp.nsum(summand, [2, n])
    term5 = -4/n * sum_term

    return term1 + term2 + term3 + term4 + term5


def c_n_1loop(n, t, mu, Nc=3):
    """
    Full one-loop coefficient.
    """
    CF = (Nc**2 - 1) / (2*Nc)

    gamma = gamma_n(n, CF)
    B = B_n(n)

    return CF * (gamma * mp.log(8*mp.pi*mu**2*t) + B)

def c_numeric(n, t, mu):
    return float(c_n_1loop(n, t, mu))

# parameters
n = 4
mu = 2.0

# flow time range (log scale is important!)
x_values = np.logspace(-4, 0, 200)
t_values = x_values / mu**2

c_values = [c_numeric(n, t, mu) for t in t_values]

plt.figure()
plt.plot(x_values, c_values)
plt.xscale("log")
plt.xlabel("μ² t")
plt.ylabel("c_n^{(1)}")
plt.show()
