# Copyright (c) 2017, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from . import LatentFunctionInference
from .posterior import HeavisidePosterior
from ...util.linalg import pdinv, dpotrs, tdot
from ...util import diag

import numpy as np
from scipy.special import gammaln, digamma


class ExactHeavisideInference(LatentFunctionInference):
    """
    An object for inference of Heaviside processes

    The function self.inference returns as HeavisidePosterior object, which
    summarizes the posterior
    """

    def inference(self, kern, X, Y, n, mean_function=None, K=None, variance=0.1):
        m = 0 if mean_function is None else mean_function.f(X)
        K = kern.K(X) if K is None else K

        YYT_factor = Y - m
        Ky = K.copy()
        diag.add(Ky, variance + 1e-8)

        # Posterior representation
        Wi, LW, LWi, W_logdet = pdinv(Ky)
        alpha, _ = dpotrs(LW, YYT_factor, lower=1)
        beta = np.sum(alpha * YYT_factor)
        posterior = HeavisidePosterior(n, woodbury_chol=LW, woodbury_vector=alpha, K=K)

        # Log marginal
        dy = Y.shape[0]
        D = Y.shape[1]
        if n == dy:  # Edge case
            n += dy * np.finfo(float).epsneg
        if beta >= n:
            log_marginal = np.full(beta.shape, np.finfo(float).min)
        else:
            log_marginal = 0.5 * (
                (n - dy) * np.log(1 - beta / n) - dy * np.log(n * np.pi)
            )
            log_marginal -= 0.5 * W_logdet
            log_marginal += gammaln(1 + n / 2) - gammaln(1 + (n - dy) / 2)

        # Gradients
        if (
            beta >= n
        ):  # Use the gradient of a Gaussian distribution to push us back towards
            # realistic values
            dL_dK = 0.5 * (tdot(alpha) - D * Wi)
            dL_dn = 10 * (
                0.5 * (digamma(1 + n / 2) - digamma(1 + (n - dy) / 2)) - dy / (2 * n)
            )
            dL_dm = alpha
        else:
            dL_dK = 0.5 * (n - dy) * tdot(alpha) / (n - beta) - 0.5 * D * Wi
            dL_dn = 0.5 * np.log(1 - beta / n) - ((n - dy) / (2 * n)) * beta / (
                n - beta
            )
            dL_dn += 0.5 * (digamma(1 + n / 2) - digamma(1 + (n - dy) / 2))
            dL_dn -= dy / (2 * n)
            dL_dm = -(n - dy) * alpha / (n - beta)
        gradients = {"dL_dK": dL_dK, "dL_dn": dL_dn, "dL_dm": dL_dm}
        # print(n, beta, dL_dn, dy, log_marginal)

        return posterior, log_marginal, gradients
