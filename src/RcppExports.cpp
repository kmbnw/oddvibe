// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// MySqrt
Rcpp::NumericVector MySqrt(const Rcpp::NumericVector& xs);
RcppExport SEXP oddvibe_MySqrt(SEXP xsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type xs(xsSEXP);
    rcpp_result_gen = Rcpp::wrap(MySqrt(xs));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"oddvibe_MySqrt", (DL_FUNC) &oddvibe_MySqrt, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_oddvibe(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
