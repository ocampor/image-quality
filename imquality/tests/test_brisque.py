import os

import PIL.Image
import pytest
import skimage.filters

from imquality.quality.brisque import Brisque
from imquality.tests import TEST_PATH
from imquality.utils import load_image

EPSILON = 0.0001


@pytest.fixture
def image() -> PIL.Image.Image:
    return load_image(os.path.join(TEST_PATH, 'resources', 'kodim05.png'))


@pytest.fixture
def brisque(image: PIL.Image.Image) -> Brisque:
    return Brisque(image, kernel_size=33, sigma=3)


def test_brisque_image_is_two_dimensions(brisque: Brisque):
    assert len(brisque.image.shape) == 2


def test_correct_local_mean(brisque: Brisque):
    expected = skimage.filters.gaussian(brisque.image, sigma=brisque.sigma, mode='constant')
    assert brisque.local_mean[:20, :20] == pytest.approx(expected[:20, :20], EPSILON)


def test_correct_local_deviation(brisque: Brisque):
    pass


def test_correct_mscn(brisque: Brisque):
    pass


def test_correct_mscn_horizontal(brisque: Brisque):
    pass


def test_correct_mscn_vertical(brisque: Brisque):
    pass


def test_correct_mscn_diagonal(brisque: Brisque):
    pass


def test_correct_mscn_secondary_diagonal(brisque: Brisque):
    pass


def test_right_amount_of_brisque_features(brisque: Brisque):
    assert len(brisque.features) == 18


def test_correct_features(brisque: Brisque):
    pass


def test_correct_score_calculation():
    pass
