import os
import pickle
from itertools import chain

import skimage.color
import numpy
import cv2
import PIL.Image
import scipy.signal
import scipy.special
import scipy.optimize
import collections

import svmutil

RESOURCES_PATH = os.path.dirname(os.path.abspath(__file__))
RESOURCES_PATH = os.path.join(RESOURCES_PATH, 'resources')


class Brisque:
    def __init__(self, image: PIL.Image):
        self.image = numpy.asarray(image)
        self.sigma = 7 / 6
        self.kernel_size = 7
        self.gray_image = skimage.color.rgb2gray(self.image)
        self.downscaled_image = cv2.resize(
            self.gray_image,
            None,
            fx=1 / 2,
            fy=1 / 2,
            interpolation=cv2.INTER_CUBIC)

    def normalize_kernel(self, kernel):
        return kernel / numpy.sum(kernel)

    def gaussian_kernel2d(self):
        n = self.kernel_size

        Y, X = numpy.indices((n, n)) - int(n / 2)
        gaussian_kernel = 1 / (2 * numpy.pi * self.sigma ** 2) * numpy.exp(-(X ** 2 + Y ** 2) / (2 * self.sigma ** 2))
        return self.normalize_kernel(gaussian_kernel)

    def local_deviation(self, image, local_mean, kernel):
        "Vectorized approximation of local deviation"
        sigma = image ** 2
        sigma = scipy.signal.convolve2d(sigma, kernel, 'same')
        return numpy.sqrt(numpy.abs(local_mean ** 2 - sigma))

    def calculate_mscn_coefficients(self, image):
        C = 1 / 255
        kernel = self.gaussian_kernel2d()
        local_mean = scipy.signal.convolve2d(image, kernel, 'same')
        local_var = self.local_deviation(image, local_mean, kernel)

        return (image - local_mean) / (local_var + C)

    def generalized_gaussian_dist(self, x, alpha, sigma):
        beta = sigma * numpy.sqrt(scipy.special.gamma(1 / alpha) / scipy.special.gamma(3 / alpha))

        coefficient = alpha / (2 * beta() * scipy.special.gamma(1 / alpha))
        return coefficient * numpy.exp(-(numpy.abs(x) / beta) ** alpha)

    def calculate_pair_product_coefficients(self, mscn_coefficients):
        return collections.OrderedDict({
            'mscn': mscn_coefficients,
            'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
            'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
            'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
            'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
        })

    def calculate_brisque_features(self, image):
        def calculate_features(coefficients_name, coefficients):
            alpha, mean, sigma_l, sigma_r = AssymmetricGeneralizedGaussian.fit(coefficients)

            if coefficients_name == 'mscn':
                var = (sigma_l ** 2 + sigma_r ** 2) / 2
                return [alpha, var]

            return [alpha, mean, sigma_l ** 2, sigma_r ** 2]

        mscn_coefficients = self.calculate_mscn_coefficients(image)
        coefficients = self.calculate_pair_product_coefficients(mscn_coefficients)

        features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
        flatten_features = list(chain.from_iterable(features))
        return numpy.array(flatten_features)

    def scale_features(self, features):
        with open(os.path.join(RESOURCES_PATH, 'normalize.pickle'), 'rb') as handle:
            scale_params = pickle.load(handle)

        min_ = numpy.array(scale_params['min_'])
        max_ = numpy.array(scale_params['max_'])

        return -1 + (2.0 / (max_ - min_) * (features - min_))

    def calculate_image_quality_score(self, brisque_features):
        model = svmutil.svm_load_model(os.path.join(RESOURCES_PATH, 'brisque_svm.txt'))
        scaled_brisque_features = self.scale_features(brisque_features)

        x, idx = svmutil.gen_svm_nodearray(
            scaled_brisque_features,
            isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))

        nr_classifier = 1
        prob_estimates = (svmutil.c_double * nr_classifier)()

        return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)

    def score(self):
        brisque_features = self.calculate_brisque_features(self.gray_image)
        downscale_brisque_features = self.calculate_brisque_features(self.downscaled_image)

        brisque_features = numpy.concatenate((brisque_features, downscale_brisque_features))

        return self.calculate_image_quality_score(brisque_features)
