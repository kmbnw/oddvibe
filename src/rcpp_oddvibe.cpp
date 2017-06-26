#include <Rcpp.h>
#include <cmath>
#include "math_x.h"
#include "booster.h"

using NumericVector = Rcpp::NumericVector;
using NumericMatrix = Rcpp::NumericMatrix;

// [[Rcpp::plugins(cpp11)]]

//' @examples
//' tmp.seed <- 1480561820
//' set.seed(tmp.seed)
//' nrows <- 50
//' nfeatures <- 2
//' intercept <- 0.75
//' beta.one <- 2.0
//' beta.two <- 5.8
//' xnoise.one <- rnorm(nrows)
//' xnoise.two <- rnorm(nrows)
//' threshold <- 0.7 * nrows * nfeatures
//'
//' # Two features
//' xs.one <- rnorm(threshold, 5.0, 1.0)
//' xs.two <- rnorm((nrows * nfeatures) - threshold, 5.0, 1.0)
//' xs <- c(beta.one * xs.one, beta.two * xs.two)
//' mat <- matrix(xs, ncol = 2)
//' ys <- intercept + beta.one * mat[, 1] + beta.two * mat[, 2]
//'
//' # add noise to avoid a boring perfect fit
//' mat[, 1] <- mat[, 1] + xnoise.one
//' mat[, 2] <- mat[, 2] + xnoise.two
//'
//' outliers <- FindOutlierWeights(mat, ys, 5000, tmp.seed)
// [[Rcpp::export]]
NumericVector FindOutlierWeights(
        const NumericMatrix& xs,
        const NumericVector& ys,
        const size_t nrounds,
        const size_t seed = 1480561820L) {
    oddvibe::Booster booster(seed);
    const auto result = booster.fit(xs, ys, nrounds);

    return Rcpp::wrap(result);
}