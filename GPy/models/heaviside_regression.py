# Copyright (c) 2017 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core import Model
from ..core.parameterization import Param
from ..core import Mapping
from ..kern import Kern, RBF
from ..inference.latent_function_inference import ExactHeavisideInference
from ..util.normalizer import Standardize
from ..util.root_find_cdf import approx_quantiles

import numpy as np
from scipy import stats, special
from paramz import ObsAr
from paramz.transformations import Logexp
from functools import partial

import warnings


class HeavisideRegression(Model):
    """
    Heaviside Process model for regression.

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf
    :param n: initial value for the shape hyperparameter n
    :param Norm normalizer: [False]

        Normalize Y with the norm given.
        If normalizer is False, no normalization will be done
        If it is None, we use GaussianNorm(alization)

    .. Note:: Multiple independent outputs are allowed using columns of Y
    """

    def __init__(
        self,
        X,
        Y,
        kernel=None,
        n=2.0,
        normalizer=None,
        mean_function=None,
        name="Heaviside process regression",
    ):
        super().__init__(name=name)
        # X
        assert X.ndim == 2
        self.set_X(X)
        self.num_data, self.input_dim = self.X.shape

        # Y
        assert Y.ndim == 2
        if normalizer is True:
            self.normalizer = Standardize()
        elif normalizer is False:
            self.normalizer = None
        else:
            self.normalizer = normalizer

        self.set_Y(Y)

        if Y.shape[0] != self.num_data:
            # There can be cases where we want inputs than outputs, for example if we have multiple latent
            # function values
            warnings.warn(
                "There are more rows in your input data X, \
                                 than in your output data Y, be VERY sure this is what you want"
            )
        self.output_dim = self.Y.shape[1]

        # Kernel
        kernel = kernel or RBF(self.X.shape[1])
        assert isinstance(kernel, Kern)
        self.kern = kernel
        self.link_parameter(self.kern)

        if self.kern._effective_input_dim != self.X.shape[1]:
            warnings.warn(
                "Your kernel has a different input dimension {} then the given X dimension {}. Be very sure this is "
                "what you want and you have not forgotten to set the right input dimenion in your kernel".format(
                    self.kern._effective_input_dim, self.X.shape[1]
                )
            )

        # Mean function
        self.mean_function = mean_function
        if mean_function is not None:
            assert isinstance(self.mean_function, Mapping)
            assert mean_function.input_dim == self.input_dim
            assert mean_function.output_dim == self.output_dim
            self.link_parameter(mean_function)

        # Shape parameter
        if n > self.num_data:
            self.nminusd = Param("n-d", float(n - self.num_data), Logexp())
        else:
            self.nminusd = Param("n-d", float(n), Logexp())
        self.link_parameter(self.nminusd)

        # Inference
        self.inference_method = ExactHeavisideInference()
        self.posterior = None
        self._log_marginal_likelihood = None

        # Insert property for plotting (not used)
        self.Y_metadata = None

    @property
    def _predictive_variable(self):
        return self.X

    def set_XY(self, X, Y):
        """
        Set the input / output data of the model
        This is useful if we wish to change our existing data but maintain the same model

        :param X: input observations
        :type X: np.ndarray
        :param Y: output observations
        :type Y: np.ndarray or ObsAr
        """
        self.update_model(False)
        self.set_Y(Y)
        self.set_X(X)
        self.update_model(True)

    def set_X(self, X):
        """
        Set the input data of the model

        :param X: input observations
        :type X: np.ndarray
        """
        assert isinstance(X, np.ndarray)
        state = self.update_model()
        self.update_model(False)
        self.X = ObsAr(X)
        self.update_model(state)

    def set_Y(self, Y):
        """
        Set the output data of the model

        :param Y: output observations
        :type Y: np.ndarray or ObsArray
        """
        assert isinstance(Y, (np.ndarray, ObsAr))
        state = self.update_model()
        self.update_model(False)
        if self.normalizer is not None:
            self.normalizer.scale_by(Y)
            self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
            self.Y = Y
        else:
            self.Y = ObsAr(Y) if isinstance(Y, np.ndarray) else Y
            self.Y_normalized = self.Y
        self.update_model(state)

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in this class this method re-performs inference, recalculating the posterior, log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, grad_dict = (
            self.inference_method.inference(
                self.kern,
                self.X,
                self.Y_normalized,
                self.nminusd + self.num_data,
                self.mean_function,
            )
        )
        self.kern.update_gradients_full(grad_dict["dL_dK"], self.X)
        if self.mean_function is not None:
            self.mean_function.update_gradients(grad_dict["dL_dm"], self.X)
        self.nminusd.gradient = grad_dict["dL_dn"]

    def log_likelihood(self):
        """
        The log marginal likelihood of the model, :math:`p(\mathbf{y})`, this is the objective function of the model being optimised
        """
        return self._log_marginal_likelihood or self.inference()[1]

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        For making predictions, does not account for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        .. math::
            p(f*|X*, X, Y) = \int^{\inf}_{\inf} p(f*|f,X*)p(f|X,Y) df

        """
        mu, var, Sigma = self.posterior._raw_predict(
            kern=self.kern if kern is None else kern,
            Xnew=Xnew,
            pred_var=self._predictive_variable,
            full_cov=full_cov,
        )
        if self.mean_function is not None:
            mu += self.mean_function.f(Xnew)
        return mu, var, Sigma

    def predict(self, Xnew, full_cov=False, kern=None, **kwargs):
        """
        Predict the function(s) at the new point(s) Xnew. For Bimodal processes, this method is equivalent to
        predict_noiseless as no likelihood is included in the model.
        """
        return self.predict_noiseless(Xnew, full_cov=full_cov, kern=kern)

    def predict_noiseless(self, Xnew, full_cov=False, kern=None):
        """
        Predict the underlying function  f at the new point(s) Xnew.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray (Nnew x self.input_dim)
        :param full_cov: whether to return the full covariance matrix, or just the diagonal
        :type full_cov: bool
        :param kern: The kernel to use for prediction (defaults to the model kern).

        :returns: (mean, var):
            mean: posterior mean, a Numpy array, Nnew x self.input_dim
            var: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise

           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim.
           If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.
        """
        # Predict the latent function values
        mu, var, _ = self._raw_predict(Xnew, full_cov=full_cov, kern=kern)

        # Un-apply normalization
        if self.normalizer is not None:
            mu, var = self.normalizer.inverse_mean(
                mu
            ), self.normalizer.inverse_variance(var)

        return mu, var

    def __cdf_function(self, y, mu, sigma, n):
        normalised_y = (y - mu) / sigma
        prefactor = special.gamma(1 + n / 2) / (
            np.sqrt(n * np.pi) * special.gamma(0.5 * (n + 1))
        )
        return 0.5 + prefactor * normalised_y * special.hyp2f1(
            0.5, 0.5 * (1 - n), 1.5, normalised_y * normalised_y / n
        )

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), kern=None, **kwargs):
        """
        Get the predictive quantiles around the prediction at X

        :param X: The points at which to make a prediction
        :type X: np.ndarray (Xnew x self.input_dim)
        :param quantiles: tuple of quantiles, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :param kern: optional kernel to use for prediction
        :type predict_kw: dict
        :returns: list of quantiles for each X and predictive quantiles for interval combination
        :rtype: [np.ndarray (Xnew x self.output_dim), np.ndarray (Xnew x self.output_dim)]
        """
        mu, _, Sigma = self._raw_predict(X, full_cov=False, kern=kern)
        std = np.sqrt(Sigma)

        approx_quantile_location = approx_quantiles(
            partial(
                self.__cdf_function, mu=mu, sigma=std, n=self.nminusd + self.num_data
            ),
            quantiles,
            x0=mu,
        )

        if self.normalizer is not None:
            approx_quantile_location = [
                self.normalizer.inverse_mean(x) for x in approx_quantile_location
            ]

        return approx_quantile_location

    def posterior_samples(
        self,
        X,
        size=10,
        full_cov=False,
        Y_metadata=None,
        likelihood=None,
        **predict_kwargs
    ):
        """
        Not implemented for Heaviside process regression
        """
        raise NotImplementedError
