import math
import typing

import tensorflow as tf
from matplotlib import pyplot as plt


def image_normalization(image: tf.Tensor, new_min=0, new_max=255) -> tf.Tensor:
    original_dtype = image.dtype
    new_min = tf.constant(new_min, dtype=tf.float32)
    new_max = tf.constant(new_max, dtype=tf.float32)
    image_min = tf.cast(tf.reduce_min(image), tf.float32)
    image_max = tf.cast(tf.reduce_max(image), tf.float32)
    image = tf.cast(image, tf.float32)

    normalized_image = (new_max - new_min) / (image_max - image_min) * (image - image_min) + new_min
    return tf.cast(normalized_image, original_dtype)


def normalize_kernel(kernel: tf.Tensor) -> tf.Tensor:
    return kernel / tf.reduce_sum(kernel)


def gaussian_kernel2d(kernel_size: int, sigma: float, dtype=tf.float32) -> tf.Tensor:
    _range = tf.range(kernel_size)
    x, y = tf.meshgrid(_range, _range)
    constant = tf.cast(tf.round(kernel_size / 2), dtype=dtype)
    x = tf.cast(x, dtype=dtype) - constant
    y = tf.cast(y, dtype=dtype) - constant
    kernel = 1 / (2 * math.pi * sigma ** 2) * tf.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(kernel)


def gaussian_filter(image: tf.Tensor, kernel_size: int, sigma: float, dtype=tf.float32) -> tf.Tensor:
    kernel = gaussian_kernel2d(kernel_size, sigma)
    original_shape = image.get_shape()
    if image.get_shape().ndims == 3:
        image = image[tf.newaxis, :, :, :]
    image = tf.cast(image, tf.float32)
    image = tf.nn.conv2d(image, kernel[:, :, tf.newaxis, tf.newaxis], strides=1, padding='SAME')
    image = tf.reshape(image, original_shape)
    return tf.cast(image, dtype)


def rescale(image: tf.Tensor, scale: float, dtype=tf.float32, **kwargs) -> tf.Tensor:
    assert image.get_shape().ndims in (3, 4), 'The tensor must be of dimension 3 or 4'

    def get_scaled_size() -> tf.Tensor:
        shape = tf.cast(tf.shape(image), tf.float32)
        shape = shape[:2] if image.get_shape().ndims == 3 else shape[1:3]
        return tf.cast(shape * scale, tf.int32)

    image = tf.cast(image, tf.float32)
    rescale_size = get_scaled_size()
    rescaled_image = tf.image.resize(image, size=rescale_size, **kwargs)
    return tf.cast(rescaled_image, dtype)


def read_image(filename: str, **kwargs) -> tf.Tensor:
    stream = tf.io.read_file(filename)
    return tf.image.decode_image(stream, **kwargs)


def show_images(images: typing.List[tf.Tensor], **kwargs):
    fig, axs = plt.subplots(1, len(images), figsize=(19, 10))
    for image, ax in zip(images, axs):
        assert image.get_shape().ndims in (3, 4), 'The tensor must be of dimension 3 or 4'
        if image.get_shape().ndims == 4:
            image = tf.squeeze(image)

        _ = ax.imshow(image, **kwargs)
        ax.axis('off')
    fig.tight_layout()
