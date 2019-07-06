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


class AssymmetricGeneralizedGaussian:
    def estimate_phi(self, alpha):
        numerator = scipy.special.gamma(2 / alpha) ** 2
        denominator = scipy.special.gamma(1 / alpha) * scipy.special.gamma(3 / alpha)
        return numerator / denominator

    def estimate_r_hat(self, x):
        size = numpy.prod(x.shape)
        return (numpy.sum(numpy.abs(x)) / size) ** 2 / (numpy.sum(x ** 2) / size)

    def estimate_R_hat(self, r_hat, gamma):
        numerator = (gamma ** 3 + 1) * (gamma + 1)
        denominator = (gamma ** 2 + 1) ** 2
        return r_hat * numerator / denominator

    def mean_squares_sum(self, x, filter=lambda z: z == z):
        filtered_values = x[filter(x)]
        squares_sum = numpy.sum(filtered_values ** 2)
        return squares_sum / (filtered_values.shape)

    def estimate_gamma(self, x):
        left_squares = self.mean_squares_sum(x, lambda z: z < 0)
        right_squares = self.mean_squares_sum(x, lambda z: z >= 0)

        return numpy.sqrt(left_squares) / numpy.sqrt(right_squares)

    def estimate_alpha(self, x):
        r_hat = self.estimate_r_hat(x)
        gamma = self.estimate_gamma(x)
        R_hat = self.estimate_R_hat(r_hat, gamma)

        solution = scipy.optimize.root(lambda z: self.estimate_phi(z) - R_hat, [0.2]).x

        return solution[0]

    def estimate_sigma(self, x, filter=lambda z: z < 0):
        return numpy.sqrt(self.mean_squares_sum(x, filter))

    def estimate_mean(self, alpha, sigma_l, sigma_r, constant):
        return (sigma_r - sigma_l) * constant * (scipy.special.gamma(2 / alpha) / scipy.special.gamma(1 / alpha))

    def fit(self, x):
        alpha = self.estimate_alpha(x)
        sigma_l = self.estimate_sigma(x, alpha, lambda z: z < 0)
        sigma_r = self.estimate_sigma(x, alpha, lambda z: z >= 0)

        constant = numpy.sqrt(scipy.special.gamma(1 / alpha) / scipy.special.gamma(3 / alpha))
        mean = self.estimate_mean(alpha, sigma_l, sigma_r, constant)

        return alpha, mean, sigma_l, sigma_r


class Brisque:
    def __init__(self, image: PIL.Image):
        self.image = numpy.asarray(image)
        self.sigma = 7 / 6
        self.kernel_size = 7
        self.gray_image = skimage.color.rgb2gray(self.image)
        self.downscaled_image = cv2.resize(self.gray_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_CUBIC)

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
