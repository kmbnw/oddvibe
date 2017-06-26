ODdVIBe is Outlier Detection via Iterated Boosting.

See https://www.researchgate.net/publication/227006463_Iterated_Boosting_for_Outlier_Detection
Also at https://link.springer.com/chapter/10.1007%2F3-540-34416-0_23#page-1

To use from R, run devtools::document() from the top level in e.g. RStudio.

To build the C++ library:
cd cpp
make clean all

Shared library output will be in lib/libboddvibe.so.  I templated some things to avoid needless copies when running in R, so you may be better off just copying the src/*.cpp and src/*.h files into your project, at least until I can get a simple facade over the templates.

To run the C++ tests:
cd cpp
make clean tests
