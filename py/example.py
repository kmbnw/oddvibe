from oddvibe import PyBooster
import numpy as np
import random

if __name__ == '__main__':
    tmp_seed = 1480561820
    random.seed(tmp_seed)
    nrows = 500
    nfeatures = 2
    intercept = 0.75
    beta_one = 2.0
    beta_two = 5.8

    xnoise_one = np.random.normal(0, 1, nrows)
    xnoise_two = 10.0 * np.random.normal(0, 1, nrows)
    threshold = int(0.7 * nrows)
    x_threshold = int(2 * threshold)

    # Two features
    xs_one = np.random.normal(5.0, 1.0, x_threshold)
    xs_two = np.random.normal(4000.3, 90.0, (nrows * nfeatures) - x_threshold)
    xs = np.concatenate([xs_one, xs_two])

    # need to use numpy matrices for the current Cython classes
    mat = xs.reshape((nrows, nfeatures))

    ys = intercept + beta_one * mat[:, 0] + beta_two * mat[:, 1]

    # Set a few of obvious outliers
    outlier_idx = [1, 11]
    ys[outlier_idx] = 1000 * ys[outlier_idx]

    # add noise to avoid a boring perfect fit
    mat[:, 0] = mat[:, 0] + xnoise_one
    mat[:, 1] = mat[:, 1] + xnoise_two

    booster = PyBooster(tmp_seed)
    weights = booster.find_outlier_weights(mat, ys, 500)