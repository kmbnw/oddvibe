#include <Rcpp.h>
#include <cmath>
#include <iostream>
#include "math_x.h"
#include "booster.h"
#include "float_matrix.h"

using NumericVector = Rcpp::NumericVector;
using NumericMatrix = Rcpp::NumericMatrix;
using DoubleMatrix = oddvibe::FloatMatrix<double>;
using DoubleVector = std::vector<double>;

// [[Rcpp::plugins(cpp11)]]

//' Use boosting to find outliers
//'
//' Call this repeatedly after removing outliers from the inputs to better find
//' outliers
//'
//' @param xs NumericMatrix of features
//' @param ys NumericVector for response variable
//' @param nrounds Number of rounds of boosting
//' @param seed Random seed to initialize boosting with
//' @return Normalized counts of training instances chosen for all rounds of
//' boosting.  The largest relative value(s) are the potential outliers.
//' For example, if the return value is \code{c(0.3, 2.3, 0.5, 6.4)}, then
//' the last row / instance is the most likely possible outlier, with the
//' second row being another possible (though less likely) outlier.
//'
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
//' threshold <- 0.7 * nrows
//' x.threshold <- 2 * threshold
//'
//' # Two features
//' xs.one <- rnorm(x.threshold, 5.0, 1.0)
//' xs.two <- rnorm((nrows * nfeatures) - x.threshold, 4000.3, 90.0)
//' xs <- c(xs.one, xs.two)
//' mat <- t(matrix(xs, nrow = 2))
//' ys <- intercept + beta.one * mat[, 1] + beta.two * mat[, 2]
//'
//' # Generate some obvious outliers every 5 rows
//' row.idx <- 1000 * (1:nrows)
//' nonzeros <- seq(1, threshold, 5)
//' row.idx[-nonzeros] <- 1
//' ys <- ys * row.idx
//'
//' # add noise to avoid a boring perfect fit
//' mat[, 1] <- mat[, 1] + xnoise.one
//' mat[, 2] <- mat[, 2] + xnoise.two
//'
//' outliers <- FindOutlierWeights(mat, ys, 5000, tmp.seed)
//'
//' # look at the top 6 possible outliers vs the "top" 6 unlikely outliers
//' df <- cbind(as.data.frame(mat), data.frame(ys, outliers))
//' names(df) <- c('X1', 'X2', 'Y', 'Weight')
//' df <- df[order(df$Weight, decreasing = TRUE), ]
//' head(df)
//' tail(df)
//' @export
// [[Rcpp::export]]
NumericVector FindOutlierWeights(
        const NumericMatrix& xs,
        const NumericVector& ys,
        const size_t nrounds,
        const size_t seed = 1480561820L) {

    oddvibe::Booster booster(seed);

    // move construct a Dataset
    const oddvibe::Dataset< DoubleMatrix, DoubleVector, double > data(
        DoubleMatrix(xs.ncol(), Rcpp::as<DoubleVector>(xs)),
        Rcpp::as<DoubleVector>(ys));

    const auto result = booster.fit(data, nrounds);

    return Rcpp::wrap(result);
}