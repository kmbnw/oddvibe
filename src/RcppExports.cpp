// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// set.seed
//' @examples //' tmp.seed <- 1480561820 //' set.seed(tmp.seed) //' nrows <- 50 //' nfeatures <- 2 //' intercept <- 0.75 //' beta.one <- 2.0 //' beta.two <- 5.8 //' xnoise.one <- rnorm(nrows) //' xnoise.two <- rnorm(nrows) //' threshold <- 0.7 * nrows * nfeatures //' //' # Two features //' xs.one <- rnorm(threshold, 5.0, 1.0) //' xs.two <- rnorm((nrows * nfeatures) - threshold, 5.0, 1.0) //' xs <- c(beta.one * xs.one, beta.two * xs.two) //' mat <- matrix(xs, ncol);
RcppExport SEXP oddvibe_set.seed(SEXP ncolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< tmp.seed) //' nrows <- 50 //' nfeatures <- 2 //' intercept <- 0.75 //' beta.one <- 2.0 //' beta.two <- 5.8 //' xnoise.one <- rnorm(nrows) //' xnoise.two <- rnorm(nrows) //' threshold <- 0.7 * nrows * nfeatures //' //' # Two features //' xs.one <- rnorm(threshold, 5.0, 1.0) //' xs.two <- rnorm((nrows * nfeatures) - threshold, 5.0, 1.0) //' xs <- c(beta.one * xs.one, beta.two * xs.two) //' mat <- matrix(xs, >::type ncol(ncolSEXP);
    rcpp_result_gen = Rcpp::wrap(set.seed(ncol));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP oddvibe_FindOutlierWeights(SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"oddvibe_set.seed", (DL_FUNC) &oddvibe_set.seed, 1},
    {"oddvibe_FindOutlierWeights", (DL_FUNC) &oddvibe_FindOutlierWeights, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_oddvibe(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
