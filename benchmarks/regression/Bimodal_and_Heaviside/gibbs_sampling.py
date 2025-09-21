import numpy as np
from scipy.special import erf, hyp2f1, gamma, gammaln
from scipy.linalg import inv, sqrtm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.preprocessing import StandardScaler

np.random.seed(422)

matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rc("text", usetex=True)


def real_function(x: np.ndarray):
    return np.sin(2 * np.pi * x)


def kernel(x: np.ndarray, y: np.ndarray | None = None, length_scale=0.2, nu=2.5):
    # return Matern(length_scale=length_scale, nu=nu)(x, y)
    return RBF(length_scale=length_scale)(x, y)


class SampleMultivariate:
    def __init__(self, μ, Σ, noise_level=0.0) -> None:
        self.μ = μ
        self.Σ = Σ
        self.num_dims = len(self.μ)

        self.μfs = [self.μ[dim] for dim in range(self.num_dims)]
        self.μys = [
            np.concatenate((self.μ[:dim], self.μ[dim + 1 :]))
            for dim in range(self.num_dims)
        ]

        self.Σffs = [self.Σ[dim, dim] for dim in range(self.num_dims)]
        temp = [
            np.concatenate((self.Σ[:dim, :], self.Σ[dim + 1 :, :]), axis=0)
            for dim in range(self.num_dims)
        ]
        self.Σfys = [sig[:, dim] for sig, dim in zip(temp, range(self.num_dims))]
        self.invΣyys = [
            inv(
                np.concatenate((sig[:, :dim], sig[:, dim + 1 :]), axis=1)
                + np.diag([noise_level] * len(sig))
            )
            for sig, dim in zip(temp, range(self.num_dims))
        ]

        self.schur_complements_list = self.schur_complements()

    def gaussian_conditional_expectations(self, current_y):
        other_samples = [
            np.concatenate((current_y[:dim], current_y[dim + 1 :]))
            for dim in range(self.num_dims)
        ]
        return [
            μf + Σfy.T @ invΣyy @ (current - μy)
            for μf, Σfy, invΣyy, current, μy in zip(
                self.μfs, self.Σfys, self.invΣyys, other_samples, self.μys
            )
        ]

    def schur_complements(self):
        return [
            Σff - Σfy.T @ invΣyy @ Σfy
            for Σff, Σfy, invΣyy in zip(self.Σffs, self.Σfys, self.invΣyys)
        ]

    def _conditional_cdf(self, y, dimension, samples): ...

    def _cdf(self, y, dimension): ...

    def _solve_cdf(self, target, cdf, x0):
        def equality(x):
            return cdf(x) - target

        sol = root_scalar(equality, x0=x0)
        return sol.root

    def _sample_cdf(self, cdf, x0):
        rand = np.random.random()

        return self._solve_cdf(rand, cdf, x0)

    def gibbs_sample(self, size=10, burnin=1000, sep=100):
        self.samples = np.zeros_like(self.μ)
        return_ = []
        for sample_num in range(int(burnin + sep * size) + 1):
            for dim in range(self.num_dims):
                gaussian_exp = self.gaussian_conditional_expectations(self.samples)[dim]
                self.samples[dim] = self._sample_cdf(
                    partial(
                        self._conditional_cdf,
                        dimension=dim,
                        samples=self.samples,
                    ),
                    gaussian_exp,
                )

            if sample_num > burnin:
                if (sample_num - burnin) % sep == 0:
                    return_.append(self.samples.reshape(-1).copy())
            if sample_num % 100 == 0:
                print(f"Running sample ", sample_num)
        return return_

    def quantiles(self, quantile=0.5):
        lower, upper = 0.5 - 0.5 * quantile, 0.5 + 0.5 * quantile
        mean = self.μ
        values = [
            [
                self._solve_cdf(
                    val,
                    partial(self._cdf, dimension=dim),
                    mean[dim],
                )
                for val in (lower, upper)
            ]
            for dim in range(self.num_dims)
        ]
        return np.array(values).reshape(-1, 2)


training_X = np.array([[0.1], [0.35], [0.67], [0.83]])
training_y = real_function(training_X)

# x_scaler = StandardScaler()
# training_X = x_scaler.fit_transform(training_X)
# y_scaler = StandardScaler()
# training_y = y_scaler.fit_transform(training_y)


class GaussianSampleMultivariate(SampleMultivariate):
    def __init__(self, μ, Σ, noise_level=0.0):
        super().__init__(μ, Σ, noise_level)

    def __cdf_function(self, y, mu, sigma):
        return 0.5 * (1.0 + erf(np.divide(y - mu, np.sqrt(2) * sigma)))

    def _conditional_cdf(self, y, dimension, samples):
        mu = self.gaussian_conditional_expectations(samples)[dimension]
        sigma = np.sqrt(self.schur_complements_list[dimension])

        return self.__cdf_function(y, mu, sigma)

    def _cdf(self, y, dimension):
        mu = self.μ[dimension]
        sigma = np.sqrt(self.Σ[dimension, dimension])
        return self.__cdf_function(y, mu, sigma)


class StudenttSampleMultivariate(SampleMultivariate):
    def __init__(self, μ, Σ, ν, df=1, noise_level=0.0) -> None:
        super().__init__(μ, Σ, noise_level)
        self.ν = ν
        self.df = df

    def __cdf_function(self, y, mu, sigma, nu):
        gammaratio = np.exp(gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu))
        prefactor = gammaratio / np.sqrt(nu * np.pi)
        normalised_y = (y - mu) / sigma
        return 0.5 + prefactor * normalised_y * hyp2f1(
            0.5, 0.5 * (nu + 1), 1.5, -normalised_y * normalised_y / nu
        )

    def _conditional_cdf(self, y, dimension, samples): ...

    def _cdf(self, y, dimension):
        mu = self.μ[dimension]
        sigma = np.sqrt(self.Σ[dimension, dimension])
        nu = self.ν
        return self.__cdf_function(y, mu, sigma, nu)


class HeavisideSampleMultivariate(SampleMultivariate):
    def __init__(self, μ, Σ, n, noise_level=0.0) -> None:
        super().__init__(μ, Σ, noise_level)
        self.n = n

    def __cdf_function(self, y, mu, sigma, n):
        gammaratio = np.exp(gammaln(0.5 * n + 1) - gammaln(0.5 * (n + 1)))
        prefactor = gammaratio / np.sqrt(n * np.pi)
        normalised_y = (y - mu) / sigma
        if normalised_y < -np.sqrt(self.n):
            return np.array([0])
        if normalised_y > np.sqrt(self.n):
            return np.array([1])
        res = 0.5 + prefactor * normalised_y * hyp2f1(
            0.5, 0.5 * (1 - n), 1.5, normalised_y * normalised_y / n
        )
        if ~np.isfinite(res):
            print(prefactor, normalised_y, n)
            exit()
        return res

    def _cdf(self, y, dimension):
        mu = self.μ[dimension]
        sigma = np.sqrt(self.Σ[dimension, dimension])
        n = self.n
        return self.__cdf_function(y, mu, sigma, n)

    def quantiles(self, quantile=0.5):
        if quantile < 1:
            return super().quantiles(quantile=quantile)
        elif quantile == 1:
            mu = self.μ.reshape(-1)
            sigma = np.sqrt(self.n * np.diag(self.Σ))
            return np.array([mu - sigma, mu + sigma]).T


class HeavisideSampleMultivariate_original(SampleMultivariate):
    def __init__(self, μ, Σ, n) -> None:
        super().__init__(μ, Σ)
        self.n = n

    def __cdf_function(self, y, mu, sigma, n):
        prefactor = gamma(0.5 * n + 1) / (np.sqrt(np.pi) * gamma(0.5 * (n - 1) + 1))
        normalised_y = (y - mu) / sigma
        return 0.5 + prefactor * normalised_y * hyp2f1(
            0.5, 0.5 * (1 - n), 1.5, normalised_y * normalised_y
        )

    def _cdf(self, y, dimension):
        mu = self.μ[dimension]
        sigma = np.sqrt(self.Σ[dimension, dimension])
        n = self.n
        return self.__cdf_function(y, mu, sigma, n)

    def quantiles(self, quantile=0.5):
        if quantile < 1:
            return super().quantiles(quantile=quantile)
        elif quantile == 1:
            mu = self.μ.reshape(-1)
            sigma = np.sqrt(np.diag(self.Σ))
            return np.array([mu - sigma, mu + sigma]).T


class BimodalSampleMultivariate(SampleMultivariate):
    def __init__(self, μ, Σ, n, noise_level=0.0) -> None:
        super().__init__(μ, Σ, noise_level)
        self.n = n
        if self.n < 3:
            self.mode_offset = np.sqrt((3 - self.n) * np.diag(self.Σ))

    def __cdf_function(self, y, mu, sigma, n):
        normalised_y = (y - mu) / sigma
        erf_term = 0.5 * (1 + erf(normalised_y / np.sqrt(2.0)))
        exp_term = (
            normalised_y
            * np.exp(-0.5 * normalised_y * normalised_y)
            / (n * np.sqrt(2 * np.pi))
        )
        return erf_term - exp_term

    def _cdf(self, y, dimension):
        mu = self.μ[dimension]
        sigma = np.sqrt(self.Σ[dimension, dimension])
        n = self.n
        return self.__cdf_function(y, mu, sigma, n)

    def __integral_within_distance_mode(self, dist, dimension):
        if self.n < 3:
            mode = self.μ[dimension] + self.mode_offset[dimension]
            return self._cdf(mode + dist, dimension) - self._cdf(mode - dist, dimension)
        else:
            raise ValueError("Mode is unique, you do not want this")

    def quantiles(self, quantile=0.5):
        if self.n < 3:  # bimodal
            # Do one half first, and see if it works
            half_quantile = 0.5 * quantile

            def integral_equal_quantile(y, dim):
                return self.__integral_within_distance_mode(y, dim) - half_quantile

            distance_containing_quantile = np.array(
                [
                    root_scalar(
                        partial(integral_equal_quantile, dim=dim), x0=0, bracket=[0, 10]
                    ).root  # [0]
                    for dim in range(self.num_dims)
                ]
            )
            if any(
                [
                    dist - offset > 0
                    for dist, offset in zip(
                        distance_containing_quantile, self.mode_offset
                    )
                ]
            ):  # quantile is too big
                return super().quantiles(quantile=quantile)
            else:
                return np.array(
                    [
                        self.μ.reshape(-1)
                        - self.mode_offset
                        - distance_containing_quantile,
                        self.μ.reshape(-1)
                        - self.mode_offset
                        + distance_containing_quantile,
                        self.μ.reshape(-1)
                        + self.mode_offset
                        - distance_containing_quantile,
                        self.μ.reshape(-1)
                        + self.mode_offset
                        + distance_containing_quantile,
                    ]
                ).T
        else:
            return super().quantiles(quantile=quantile)
