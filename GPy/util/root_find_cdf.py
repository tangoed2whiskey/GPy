"""
Find the approximate quantiles of a given CDF using root-finding
"""

from scipy.optimize import root_scalar
from functools import partial


def approx_quantiles(cdf, quantiles: tuple[float | int] = (2.5, 97.5), x0: float = 0):
    """
    Take a CDF and a tuple of quantiles, returning the x values at
    which the CDF achieves those quantiles
    """

    def equality(x, q):
        return cdf(x) - q

    locations = [root_scalar(partial(equality, q=q), x0=x0).root for q in quantiles]

    return locations
