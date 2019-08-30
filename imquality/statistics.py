from enum import Enum
from typing import Callable

import numpy
import scipy.optimize
import scipy.special


class DistributionSide(Enum):
    right = 1
    left = 2


def find_root(fun: Callable, x0: float = 0.2) -> scipy.optimize.OptimizeResult:
    x0 = numpy.array([x0])
    return scipy.optimize.root(fun, x0)


class AsymmetricGeneralizedGaussian:
    def __init__(self, x: numpy.ndarray):
        assert isinstance(x, numpy.ndarray)

        self.x = x
        self._alpha = None

    def _x(self, side: DistributionSide):
        if side == DistributionSide.left:
            idx = numpy.where(self.x < 0)
        elif side == DistributionSide.right:
            idx = numpy.where(self.x >= 0)
        else:
            raise ValueError(f'Side {side} was not recognized')

        return self.x[idx]

    @property
    def x_left(self):
        return self._x(DistributionSide.left)

    @property
    def x_right(self):
        return self._x(DistributionSide.right)

    def _sigma(self, side: DistributionSide):
        if side == DistributionSide.right:
            _x = self.x_right
        elif side == DistributionSide.left:
            _x = self.x_left
        else:
            raise ValueError(f'Side {side} was not recognized')

        return numpy.sqrt(self.mean_squares(_x))

    @property
    def sigma_left(self):
        return self._sigma(DistributionSide.left)

    @property
    def sigma_right(self):
        return self._sigma(DistributionSide.right)

    @property
    def gamma(self):
        return self.sigma_left / self.sigma_right

    @property
    def r_hat(self):
        return numpy.abs(self.x).mean() ** 2 / self.mean_squares(self.x)

    @property
    def R_hat(self):
        return self.r_hat * (self.gamma ** 3 + 1) * (self.gamma + 1) / (self.gamma ** 2 + 1) ** 2

    @property
    def constant(self):
        return numpy.sqrt(scipy.special.gamma(1 / self.alpha) / scipy.special.gamma(3 / self.alpha))

    @property
    def mean(self):
        return (self.sigma_right - self.sigma_left) * self.constant * (
                scipy.special.gamma(2 / self.alpha) / scipy.special.gamma(1 / self.alpha))

    @property
    def alpha(self):
        if self._alpha is None:
            raise NotImplementedError('The distribution has no alpha estimated. Run method fit() to calculate.')
        return self._alpha

    @staticmethod
    def mean_squares(x: numpy.array):
        return numpy.square(x).mean()

    @staticmethod
    def phi(alpha):
        return (scipy.special.gamma(2 / alpha) ** 2 /
                (scipy.special.gamma(1 / alpha) * scipy.special.gamma(3 / alpha)))

    def estimate_alpha(self, x0: float = 0.2) -> float:
        try:
            solution = find_root(lambda alpha: self.phi(alpha) - self.R_hat, x0)
            assert solution.success
            return solution.x.item()
        except ValueError:
            raise ValueError(f'More than one solution was found for phi(alpha) - {self.R_hat}')

    def fit(self, x0: float = 0.2) -> 'AsymmetricGeneralizedGaussian':
        self._alpha = self.estimate_alpha(x0)
        return self


def normalize_kernel(kernel: numpy.ndarray) -> numpy.ndarray:
    return kernel / numpy.sum(kernel)


def gaussian_kernel2d(kernel_size, sigma: float):
    y, x = numpy.indices((kernel_size, kernel_size)) - int(kernel_size / 2)
    kernel = 1 / (2 * numpy.pi * sigma ** 2) * numpy.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(kernel)
