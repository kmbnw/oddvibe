#include <Rcpp.h>
#include <cmath>
#include "math_x.h"
#include "booster.h"

using NumericVector = Rcpp::NumericVector;
using NumericMatrix = Rcpp::NumericMatrix;

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
//' threshold <- 0.7 * nrows * nfeatures
//'
//' # Two features
//' xs.one <- rnorm(threshold, 5.0, 1.0)
//' xs.two <- rnorm((nrows * nfeatures) - threshold, 5.0, 1.0)
//' xs <- c(beta.one * xs.one, beta.two * xs.two)
//' mat <- matrix(xs, ncol = 2)
//' ys <- intercept + beta.one * mat[, 1] + beta.two * mat[, 2]
//'
//' # Generate some obvious outliers every 5 rows
//' row.idx <- 1000 * (1:nrows)
//' nonzeros <- seq(1, length(row.idx), 5)
//' row.idx[-nonzeros] <- 1
//' ys <- ys * row.idx
//'
//' # add noise to avoid a boring perfect fit
//' mat[, 1] <- mat[, 1] + xnoise.one
//' mat[, 2] <- mat[, 2] + xnoise.two
//'
//' outliers <- FindOutlierWeights(mat, ys, 5000, tmp.seed)
//' print (max(outliers))
//' # 8.662468
//' print (which(max(outliers) == outliers))
//' # 43
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