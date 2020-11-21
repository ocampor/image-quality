import os

import PIL.Image
import numpy
import pytest

from imquality.brisque import Brisque, score
from imquality.utils import load_image
from tests import TEST_PATH

EPSILON = 0.0001


@pytest.fixture
def image() -> PIL.Image.Image:
    return load_image(os.path.join(TEST_PATH, "resources", "kodim05.png"))


@pytest.fixture
def brisque(image: PIL.Image.Image) -> Brisque:
    return Brisque(image, kernel_size=7, sigma=7 / 6)


def test_brisque_image_is_two_dimensions(brisque: Brisque):
    assert len(brisque.image.shape) == 2


def test_correct_local_mean(brisque: Brisque):
    expected = numpy.array(
        [
            [0.17496204, 0.23679088, 0.25726582],
            [0.23680395, 0.32046324, 0.3480735],
            [0.25734442, 0.34822976, 0.37807359],
        ]
    )
    assert expected == pytest.approx(brisque.local_mean[:3, :3], EPSILON)


def test_correct_local_deviation(brisque: Brisque):
    expected = numpy.array(
        [
            [0.19316824, 0.1893495, 0.18346567],
            [0.18935987, 0.14730075, 0.11782558],
            [0.18352106, 0.11787542, 0.06051347],
        ]
    )
    assert expected == pytest.approx(brisque.local_deviation[:3, :3], EPSILON)


def test_correct_mscn(brisque: Brisque):
    expected = numpy.array(
        [
            [1.08211202, 0.78358554, 0.69892418],
            [0.78347589, 0.44816172, 0.32987871],
            [0.69829835, 0.32846076, 0.15770463],
        ]
    )
    assert expected == pytest.approx(brisque.mscn[:3, :3], EPSILON)


def test_correct_mscn_horizontal(brisque: Brisque):
    expected = numpy.array(
        [
            [0.84792733, 0.54766688, 0.48011944],
            [0.3511239, 0.14783901, 0.10333125],
            [0.2293636, 0.05179978, 0.01466904],
        ]
    )
    assert expected == pytest.approx(brisque.mscn_horizontal[:3, :3], EPSILON)


def test_correct_mscn_vertical(brisque: Brisque):
    expected = numpy.array(
        [
            [0.84780868, 0.35117304, 0.23056021],
            [0.54709992, 0.14720354, 0.0520234],
            [0.47840159, 0.10102396, 0.01848178],
        ]
    )
    assert expected == pytest.approx(brisque.mscn_vertical[:3, :3], EPSILON)


def test_correct_mscn_diagonal(brisque: Brisque):
    expected = numpy.array(
        [
            [0.48496118, 0.25848819, 0.2189311],
            [0.25734108, 0.07067718, 0.03068398],
            [0.2147741, 0.0384931, 0.04329395],
        ]
    )
    assert expected == pytest.approx(brisque.mscn_diagonal[:3, :3], EPSILON)


def test_correct_mscn_secondary_diagonal(brisque: Brisque):
    expected = numpy.array(
        [
            [0.61392038, 0.31323106, 0.2266071],
            [0.31295059, 0.10835221, 0.04939942],
            [0.22502724, 0.04850487, 0.01090076],
        ]
    )
    assert expected == pytest.approx(brisque.mscn_secondary_diagonal[:3, :3], EPSILON)


def test_right_amount_of_brisque_features(brisque: Brisque):
    assert len(brisque.features) == 18


def test_correct_features(brisque: Brisque):
    expected = numpy.array(
        [
            2.5831384826064765,
            0.3246368578871716,
            0.804777755330943,
            0.08897788603336909,
            0.06840297824679546,
            0.15626675254070954,
            0.8049535099622015,
            0.07516229246587908,
            0.07504565998142923,
            0.14971732887316655,
            0.7375110596615735,
            -0.010579280060541355,
            0.13411853419378322,
            0.12239672859496961,
            0.7504928455135851,
            -0.007148436867571834,
            0.1252422296198859,
            0.11758092662669493,
        ]
    )
    assert brisque.features == pytest.approx(expected, EPSILON)


@pytest.mark.parametrize(
    "file_path,expected",
    [
        ("img151.bmp", 14.2005),
        ("img16.bmp", 63.4818),
        ("img176.bmp", 89.9085),
        ("kodim05.png", 4.22978),
    ],
)
def test_correct_score_calculation(file_path, expected):
    image = load_image(os.path.join(TEST_PATH, "resources", file_path))
    assert score(image) == pytest.approx(expected, 3)
