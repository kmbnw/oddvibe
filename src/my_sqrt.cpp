#include <Rcpp.h>
#include <cmath>

// [[Rcpp::plugins(cpp14)]]

inline static double sqrt_double(double x) {
    return ::sqrt(x);
}

// [[Rcpp::export]]
Rcpp::NumericVector
MySqrt(const Rcpp::NumericVector& xs) {
    Rcpp::NumericVector result(xs.size());

    std::transform(
        xs.begin(),
        xs.end(),
        result.begin(),
        sqrt_double);

    return result;
}