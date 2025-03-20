import warnings
import numpy as np
from scipy.linalg import svd
from scipy.optimize import OptimizeResult, OptimizeWarning
from scipy.stats.distributions import t, f
from typing import Callable, List


def get_weights(std_x: [np.ndarray, float], std_y: [np.ndarray, float] = 0) -> np.ndarray:
    """
    Creates a vector of weights considering the standard deviation on the x and y data
    :param std_x: The standard deviation of the x data
    :type np.ndarray
    :param std_y: The standard deviation of the y data
    :type np.ndarray
    :return: An array with the weights for the minimization
    :rtype np.ndarray
    """
    s2 = std_x ** 2. + std_y ** 2.
    # Add a constant to avoid division by zero
    s2 += np.median(std_x)
    return 1. / s2


def ls_covariance(ls_res: OptimizeResult, absolute_sigma=False):
    """
    Estimates the covariance matrix for a `scipy.optimize.least_squares` result
    :param ls_res: The object returned by `scipy.optimize.least_squares`
    :type ls_res: OptimizeResult
    :param absolute_sigma: If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.

        If False (default), only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit. Default is False.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    :type absolute_sigma: bool
    :return: The covariance matrix of the fit
    :rtype np.ndarray
    """
    popt = ls_res.x
    ysize = len(ls_res.fun)
    cost = 2. * ls_res.cost  # res.cost is half sum of squares!

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(ls_res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(ls_res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s ** 2, VT)

    if pcov is None or np.isnan(pcov).any():
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
        warnings.warn('Covariance of the parameters could not be estimated.',
                      category=OptimizeWarning)
    elif not absolute_sigma:
        if ysize > len(popt):
            s_sq = cost / (ysize - len(popt))
            pcov = pcov * s_sq
    return pcov


def confidence_interval(ls_res: OptimizeResult, level: float = 0.95, absolute_sigma=False):
    """
    Estimates the confidence interval for the parameters fitted using
    scipy's `least_squares` function for a given confidence level
    (default is 95%).

    :param ls_res: The result returned by `least_squares`.
    :type ls_res: OptimizeResult
    :param level: The confidence level.
    :type level: float, optional
    :param absolute_sigma: If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.

        If False (default), only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit. Default is False.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    :type absolute_sigma: bool
    :return: A numpy array with containing the confidence intervals for
        each parameter.
    :rtype: np.ndarray
    """
    if not isinstance(ls_res, OptimizeResult):
        msg = 'Expecting \'ls_res\' an instance of \'scipy.optimize.OptimizeResult\' for \'ls_res\''
        raise ValueError(msg)

    pcov = ls_covariance(ls_res, absolute_sigma=absolute_sigma)
    # The vector of residuals at the solution
    residuals = ls_res.fun
    # The number of data points
    n = len(residuals)
    # The number of parameters
    p = len(ls_res.x)
    # The degrees of freedom
    dof = n - p

    # Quantile of Student's t distribution for p=(1 - alpha/2)
    # tval = t.ppf((1.0 + confidence)/2.0, dof)
    alpha = 1.0 - level
    tval = t.ppf(1.0 - alpha / 2.0, dof)

    ci = np.zeros((p, 2), dtype=np.float64)

    for i, p, var in zip(range(n), ls_res.x, np.diag(pcov)):
        sigma = var ** 0.5
        ci[i, :] = [p - sigma * tval, p + sigma * tval]

    return ci


def prediction_intervals(model: Callable, x_pred, ls_res: OptimizeResult, level=0.95,
                         jac: Callable = None, weights: np.ndarray = None, **kwargs):
    """
    Estimates the prediction interval for a least` squares fit result obtained by
    scipy.optimize.least_squares.

    :param model: The model used to fit the data
    :type model: Callable
    :param x_pred: The values of X at which the model will be evaluated.
    :type x_pred: np.ndarray
    :param ls_res: The result object returned by scipy.optimize.least_squares.
    :type ls_res: OptimizeResult
    :param level: The confidence level used to determine the prediction intervals.
    :type level: float
    :param jac: The Jacobian of the model at the parameters. If not provided,
        it will be estimated from the model. Default None.
    :type jac: Callable
    :param weights: The weights of the datapoints used for the fitting.
    :type weights: np.ndarray
    :param kwargs:
    :return: The predicted values at the given x and the deltas for each prediction
        [y_predicction, delta]
    :rtype: List[np.ndarray, np.ndarray]
    """

    simultaneous = kwargs.get('simultaneous', False)
    new_observation = kwargs.get('new_observation', False)

    # The vector of residuals at the solution
    residuals = ls_res.fun
    beta = ls_res.x
    # The number of data points
    n = len(residuals)
    # The number of parameters
    p = len(beta)
    if n <= p:
        raise ValueError('Not enough data to compute the prediction intervals.')
    # The degrees of freedom
    dof = n - p
    # Quantile of Student's t distribution for p=(1 - alpha/2)
    # tval = t.ppf((1.0 + confidence)/2.0, dof)
    alpha = 1.0 - level

    # Compute the predicted values at the new x_pred
    y_pred = model(x_pred, ls_res.x)
    delta = np.empty((len(y_pred), p), dtype=np.float64)
    fdiffstep = np.finfo(beta.dtype).eps ** (1. / 3.)
    if jac is None:
        for i in range(p):
            change = np.zeros(p)
            if beta[i] == 0:
                nb = np.linalg.norm(beta)
                change[i] = fdiffstep * (nb + float(nb == 0))
            else:
                change[i] = fdiffstep * beta[i]
            predplus = model(x_pred, beta + change)
            delta[:, i] = (predplus - y_pred) / change[i]
    else:
        delta = jac(beta, x_pred, y_pred)

    J = ls_res.jac

    # Find R to get the variance
    _, R = np.linalg.qr(J)
    # Get the rank of jac_pnp
    rankJ = J.shape[1]
    Rinv = np.linalg.pinv(R)
    pinvJTJ = np.dot(Rinv, Rinv.T)

    # Get MSE. The degrees of freedom when J is full rank is v = n-p and n-rank(J) otherwise
    mse = (np.linalg.norm(residuals)) ** 2. / (n - rankJ)

    # Calculate Sigma if usingJ
    sigma = mse * pinvJTJ

    # Compute varpred
    varpred = np.sum(np.dot(delta, sigma) * delta, axis=1)

    if new_observation:
        if not weights is None:
            error_var = mse / weights
        else:
            error_var = mse * np.ones(delta.shape[0])
        varpred += error_var

    if simultaneous:
        if new_observation:
            sch = [rankJ + 1]
        else:
            sch = rankJ
        crit = np.sqrt(sch * (f.ppf(1.0 - alpha, sch, n - rankJ)))
    else:
        from scipy.stats.distributions import t
        crit = t.ppf(1.0 - alpha / 2.0, n - rankJ)

    delta = np.sqrt(varpred) * crit

    return y_pred, delta
