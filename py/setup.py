from distutils.core import setup, Extension
from Cython.Build import cythonize
from glob import glob

# advice for C++ 11 setup from
# https://stackoverflow.com/questions/27305343/cython-not-recognizing-c11-commands

extensions = [
    Extension(
        name='oddvibe',
        sources=list(
            set(glob('*.pyx') + glob('../src/*.cpp')) -
            set(['../src/rcpp_oddvibe.cpp', '../src/RcppExports.cpp'])),
        language='c++',
        extra_compile_args=['-std=c++11']
    )
]

setup(
    name = "oddvibe",
    ext_modules = cythonize(extensions),
)