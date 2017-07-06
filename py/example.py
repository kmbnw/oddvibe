from oddvibe import PyBooster
import numpy as np
import random

if __name__ == '__main__':
    tmp_seed = 1480561820
    random.seed(tmp_seed)
    nrows = 50
    nfeatures = 2
    intercept = 0.75
    beta_one = 2.0
    beta_two = 5.8

    xnoise_one = np.random.normal(0, 1, nrows)
    xnoise_two = np.random.normal(0, 1, nrows)
    threshold = int(0.7 * nrows)
    x_threshold = int(2 * threshold)

    # Two features
    xs_one = np.random.normal(5.0, 1.0, x_threshold)
    xs_two = np.random.normal(4000.3, 90.0, (nrows * nfeatures) - x_threshold)
    xs = np.concatenate([xs_one, xs_two])

    mat = xs.reshape((nrows, nfeatures))

    ys = intercept + beta_one * mat[:, 0] + beta_two * mat[:, 1]

    # Generate some obvious outliers every 5 rows
    row_idx = 1000 * np.array(list(xrange(1, nrows + 1)))
    nonzeros = set(list(xrange(0, threshold, 5)))
    zeros = [x for x in xrange(0, nrows) if x not in nonzeros]
    row_idx[zeros] = 1
    ys = ys * row_idx

    # add noise to avoid a boring perfect fit
    mat[:, 0] = mat[:, 0] + xnoise_one
    mat[:, 1] = mat[:, 1] + xnoise_two

    booster = PyBooster(tmp_seed)
    weights = booster.find_outlier_weights(mat, ys, 5000)