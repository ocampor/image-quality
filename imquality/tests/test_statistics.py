import numpy

from imquality.statistics import AsymmetricGeneralizedGaussian


def test_asymmetric_generalized_gaussian_creation():
    data = numpy.array([0.1, 0.2, 0.3])
    AsymmetricGeneralizedGaussian(data)
