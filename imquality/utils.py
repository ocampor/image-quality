import typing
from os import PathLike

import PIL.Image
import numpy


def load_image(path: typing.Union[bytes, str, PathLike]):
    image = PIL.Image.open(path)
    # NOTE: PI.Image.open is a lazy evaluation. This line is needed to force to load content.
    image.load()
    return image


def pil2ndarray(image: PIL.Image.Image) -> numpy.ndarray:
    return numpy.asarray(image)
