import fittingtools as fls
import numpy as np
from numpy.random import default_rng
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

rng = default_rng()


def gen_data(t, a, b, c, noise=0., n_outliers=0, seed=None):
    rng = default_rng(seed)

    y = a + b * np.exp(t * c)

    error = noise * rng.standard_normal(t.size)
    outliers = rng.integers(0, t.size, n_outliers)
    error[outliers] *= 10

    return y + error


def model(t, b):
    return b[0] + b[1] * np.exp(b[2] * t)


def res(b, t, y):
    return model(t, b) - y


def jac(b, t, y):
    n, p = len(t), len(b)
    j = np.empty((n, p), dtype=np.float64)
    j[:, 0] = np.ones(n, dtype=np.float64)
    e = np.exp(b[2] * t)
    j[:, 1] = e
    j[:, 2] = b[1] * t * e
    return j


def main():
    a = 0.5
    b = 2.0
    c = -1
    t_min = 0
    t_max = 10
    n_points = 15

    t_train = np.linspace(t_min, t_max, n_points)
    y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)

    x0 = np.array([1.0, 1.0, 0.0])
    res_lsq = least_squares(res, x0, args=(t_train, y_train), loss='cauchy', f_scale=0.1, )

    t_test = np.linspace(t_min, t_max, n_points * 10)
    y_true = gen_data(t_test, a, b, c)

    popt = res_lsq.x
    parameters_ci = fls.confidence_interval(ls_res=res_lsq, level=0.95)
    for i, p, lci, uci in zip(range(len(popt)), popt, parameters_ci[:, 0], parameters_ci[:, 1]):
        print(f'beta[{i}]: {p:>7.3f}, 95% CI: [{lci:>7.3f}, {uci:>7.3f}]')

    y_pred, delta = fls.prediction_intervals(
        model=model, x_pred=t_test, ls_res=res_lsq, jac=jac
    )

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(4., 3.)
    ax.plot(t_train, y_train, 'o')
    ax.fill_between(t_test, y_pred - delta, y_pred + delta, color='C0', alpha=0.5, label='95% intervals')
    ax.plot(t_test, y_true, 'k', linewidth=2, label='true')
    ax.plot(t_test, y_pred, linewidth=2, label='cauchy')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best', frameon=True)
    fig.savefig('prediction_intervals_cauchy.png')
    plt.show()


if __name__ == '__main__':
    main()
