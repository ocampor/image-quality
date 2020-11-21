import numpy
import pytest

import imquality.statistics

EXPECTED_ALPHA = 1.32418

EPSILON = 0.00001

DATA = numpy.array([0.1, 0.2, 0.3, 0.2, -0.5, -1.3, 1.5])


@pytest.fixture
def agg():
    return imquality.statistics.AsymmetricGeneralizedGaussian(DATA)


@pytest.fixture
def initialized_agg(agg):
    return agg.fit(0.2)


def test_asymmetric_generalized_gaussian_creation(agg):
    assert agg.x == pytest.approx(DATA, EPSILON)


def test_alpha_not_initialized(agg):
    with pytest.raises(NotImplementedError):
        assert agg.alpha


def test_alpha_initialized(initialized_agg):
    assert initialized_agg.alpha


def test_left_distribution(agg):
    assert agg.x_left == pytest.approx(numpy.array([-0.5, -1.3]), EPSILON)


def test_right_distribution(agg):
    assert agg.x_right == pytest.approx(numpy.array([0.1, 0.2, 0.3, 0.2, 1.5]), EPSILON)


def test_mean_squares():
    assert imquality.statistics.AsymmetricGeneralizedGaussian.mean_squares(
        DATA
    ) == pytest.approx(0.6242857142857143, EPSILON)


def test_r_hat(agg):
    assert agg.r_hat == pytest.approx(0.5495259888852567, EPSILON)


def test_R_hat(agg):
    assert agg.R_hat == pytest.approx(0.5642625606630997, EPSILON)


def test_sigma_left(agg):
    assert agg.sigma_left == pytest.approx(0.9848857801796105, EPSILON)


def test_sigma_right(agg):
    assert agg.sigma_right == pytest.approx(0.6971370023173351, EPSILON)


def test_gamma(agg):
    assert agg.gamma == pytest.approx(1.4127578609452334, EPSILON)


@pytest.mark.parametrize(
    "alpha,expected",
    [
        (1.2, 0.543103),
        (3, 0.684463405979725),
        (1.5515093216386946, 0.5951306268698388),
    ],
)
def test_phi(alpha, expected):
    assert imquality.statistics.AsymmetricGeneralizedGaussian.phi(
        alpha
    ) == pytest.approx(expected, EPSILON)


def test_estimate_alpha(agg):
    assert agg.estimate_alpha(0.2) == pytest.approx(EXPECTED_ALPHA, EPSILON)


def test_fit(agg):
    agg = agg.fit()
    assert agg.alpha == pytest.approx(EXPECTED_ALPHA, EPSILON)


def test_constant(initialized_agg):
    assert initialized_agg.constant == pytest.approx(1.03244, EPSILON)


def test_mean(initialized_agg):
    assert initialized_agg.mean == pytest.approx(-0.216150, EPSILON)


def test_gaussian_kernel2d():
    kernel = imquality.statistics.gaussian_kernel2d(3, 3)
    expected = numpy.array(
        [
            [0.107035, 0.113092, 0.107035],
            [0.113092, 0.119491, 0.113092],
            [0.107035, 0.113092, 0.107035],
        ]
    )
    assert kernel == pytest.approx(expected, 0.001)
