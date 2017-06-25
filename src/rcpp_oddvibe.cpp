#include <Rcpp.h>
#include <cmath>
#include "math_x.h"

// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
Rcpp::NumericVector
MyRMSELoss(const Rcpp::NumericVector& ys, const Rcpp::NumericVector& yhats) {
    if (ys.size() != yhats.size()) {
        throw std::logic_error("Observed and predicted must be same size");
    }
    Rcpp::NumericVector loss(yhats.size(), 0);
    std::transform(
        yhats.begin(),
        yhats.end(),
        ys.begin(),
        loss.begin(),
        oddvibe::rmse_loss);
    return loss;
}