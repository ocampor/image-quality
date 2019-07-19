import numpy
import scipy.optimize
import scipy.special


class AsymmetricGeneralizedGaussian:
    def __init__(self, x: numpy.array):
        self.x = x
        self._alpha = None

    @property
    def gamma(self):
        left_squares = self.mean_squares(numpy.where(self.x < 0))
        right_squares = self.mean_squares(numpy.where(self.x >= 0))
        return numpy.sqrt(left_squares) / numpy.sqrt(right_squares)

    @property
    def R_hat(self):
        return self.r_hat * (self.gamma ** 3 + 1) * (self.gamma + 1) / (self.gamma ** 2 + 1) ** 2

    @property
    def r_hat(self):
        size = numpy.prod(self.x.shape)
        return (numpy.sum(numpy.abs(self.x)) / size) ** 2 / (numpy.sum(self.x ** 2) / size)

    def _sigma(self, side: str):
        if side == 'right':
            return self.mean_squares(numpy.where(self.x >= 0))
        elif side == 'left':
            return self.mean_squares(numpy.where(self.x < 0))
        else:
            raise ValueError('Side {0} was not recognized'.format(side))

    @property
    def sigma_left(self):
        return self._sigma('left')

    @property
    def sigma_right(self):
        return self._sigma('right')

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
                scipy.special.gamma(1 / alpha) * scipy.special.gamma(3 / alpha))

    def estimate_alpha(self, x0: float = 0.2):
        x0 = numpy.array([x0])
        # TODO: Verify if [0] can be removed from this expression
        return scipy.optimize.root(lambda alpha: self.phi(alpha) - self.R_hat, x0).x[0]

    def fit(self, x0: float = 0.2):
        self._alpha = self.estimate_alpha(self.x, x0)
