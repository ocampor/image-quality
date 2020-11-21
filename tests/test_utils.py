import os

import PIL.Image
import pytest

from imquality.utils import load_image, pil2ndarray
from tests import TEST_PATH


@pytest.fixture
def image() -> PIL.Image.Image:
    return load_image(os.path.join(TEST_PATH, "resources", "kodim05.png"))


def test_load_image(image: PIL.Image.Image):
    assert (image.width, image.height) == (768, 512)


def test_pil2ndarray(image: PIL.Image.Image):
    array = pil2ndarray(image)
    assert tuple(array.shape) == (512, 768, 3)
