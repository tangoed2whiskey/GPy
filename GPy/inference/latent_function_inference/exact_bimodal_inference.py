# Copyright (c) 2017, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from . import LatentFunctionInference
from .posterior import BimodalPosterior
from ...util.linalg import pdinv, dpotrs, tdot
from ...util import diag

import numpy as np
from scipy.special import gammaln, digamma


class ExactBimodalInference(LatentFunctionInference):
    """
    An object for inference of Bimodal processes

    The function self.inference returns as BimodalPosterior object, which
    summarizes the posterior
    """

    def inference(self, kern, X, Y, n, mean_function=None, K=None):
        m = 0 if mean_function is None else mean_function.f(X)
        K = kern.K(X) if K is None else K

        YYT_factor = Y - m
        Ky = K.copy()
        diag.add(Ky, 1e-8)

        # Posterior representation
        Wi, LW, LWi, W_logdet = pdinv(Ky)
        alpha, _ = dpotrs(LW, YYT_factor, lower=1)
        beta = np.sum(alpha * YYT_factor)
        posterior = BimodalPosterior(n, woodbury_chol=LW, woodbury_vector=alpha, K=K)

        # Log marginal
        dy = Y.shape[0]
        D = Y.shape[1]
        log_marginal = -0.5 * (np.log(2 * np.pi) + W_logdet + beta)
        log_marginal += np.log(1 + beta / (n - dy))
        log_marginal += np.log(n - dy) - np.log(n)

        # Gradients
