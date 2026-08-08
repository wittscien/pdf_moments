import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt


# Set precision if needed
mp.mp.dps = 15


def harmonic_number(n):
    return mp.nsum(lambda k: 1/mp.mpf(k), [1, n])

def gamma_n(n):
    Hn = harmonic_number(n)
    return 1 + 4*(Hn - 1) - 2/(n*(n+1))

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
        return (1/(j*(j-1)*2**j)) * mp.lerchphi(1/2, 1, j)

    sum_term = mp.nsum(summand, [2, n])
    term5 = -4 * sum_term

    return term1 + term2 + term3 + term4 + term5


def c_n_1loop(n, t, mu, Nc=3):
    """
    Full one-loop coefficient.
    """
    CF = (Nc**2 - 1) / (2*Nc)

    gamma = gamma_n(n)
    B = B_n(n)

    return CF * (gamma * mp.log(8*mp.pi*mu**2*t) + B)


def c_n(n, t, mu):
    alpha_s = 0.3069
    c1 = c_n_1loop(n, t, mu)
    return 1 + alpha_s/(4*mp.pi) * c1


def c_numeric(n, t, mu):
    return float(c_n(n, t, mu))


if __name__ == '__main__':
    # parameters
    n = 2
    mu = 2.0

    # flow time range (log scale is important!)
    x_values = np.linspace(0.1, 0.6) # sqrt(8t) in fm
    t_values = x_values ** 2 / 8 / 0.197 ** 2 # GeV-2
    c_values = [c_numeric(n, t, mu) for t in t_values]

    plt.figure()
    plt.plot(x_values, c_values)
    plt.xlabel(r'$\sqrt{8t}\,/\,\mathrm{fm}$')
    plt.ylabel(r'$c_{2}$')
    plt.show()
