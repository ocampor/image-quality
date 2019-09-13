import math

import tensorflow as tf
from matplotlib import pyplot as plt


@tf.function
def image_normalization(image: tf.Tensor, min=0, max=255) -> tf.Tensor:
    image_min = tf.reduce_min(image)
    image_max = tf.reduce_max(image)
    return (max - min) / (image_max - image_min) * (image - image_min) + min


@tf.function
def normalize_kernel(kernel: tf.Tensor) -> tf.Tensor:
    return kernel / tf.reduce_sum(kernel)


@tf.function
def gaussian_kernel2d(kernel_size: int, sigma: float, dtype=tf.float32) -> tf.Tensor:
    _range = tf.range(kernel_size)
    x, y = tf.meshgrid(_range, _range)
    constant = tf.cast(tf.round(kernel_size / 2), dtype=dtype)
    x = tf.cast(x, dtype=dtype) - constant
    y = tf.cast(y, dtype=dtype) - constant
    kernel = 1 / (2 * math.pi * sigma ** 2) * tf.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(kernel)


@tf.function
def gaussian_filter(image: tf.Tensor, kernel_size: int, sigma: float) -> tf.Tensor:
    kernel = gaussian_kernel2d(kernel_size, sigma)
    if image.get_shape().ndims == 3:
        image = image[tf.newaxis, :, :, :]

    return tf.nn.conv2d(image, kernel[:, :, tf.newaxis, tf.newaxis], strides=1, padding='SAME')


def show_images(images: list, **kwargs):
    fig, axs = plt.subplots(1, len(images), figsize=(19, 10))
    for image, ax in zip(images, axs):
        _ = ax.imshow(image, **kwargs)
        ax.axis('off')
    fig.tight_layout()
