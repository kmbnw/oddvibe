#
# Copyright 2016-2017 Krysta M Bouzek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# I used the following as a guide to create this file:
# http://www.birving.com/blog/2014/05/13/passing-numpy-arrays-between-python-and/
# http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html

from libcpp.vector cimport vector

cdef extern from "../src/float_matrix.h" namespace "oddvibe":
    cdef cppclass FloatMatrix "oddvibe::FloatMatrix<float>":
        FloatMatrix(size_t ncols, vector[float] xs) except +

cdef extern from "../src/dataset.h" namespace "oddvibe":
    cdef cppclass Dataset "oddvibe::Dataset<float>":
        Dataset(FloatMatrix mat, vector[float] ys) except +

cdef extern from "../src/booster.h" namespace "oddvibe":
    cdef cppclass Booster:
        Booster(size_t seed) except +
        vector[float] fit_counts(Dataset data, size_t nrounds)

cdef class PyBooster:
    cdef size_t seed

    def __cinit__(self, size_t seed):
        self.seed = seed

    def find_outlier_weights(self, xs, ys, size_t nrounds):
        cdef Booster *booster = NULL
        cdef Dataset *data = NULL
        cdef FloatMatrix *mat = NULL

        try:
            booster = new Booster(self.seed)
            mat = new FloatMatrix(xs.shape[1], xs.flatten(order = 'F'))

            # recall that [0] is for dereferencing the pointer
            data = new Dataset(mat[0], ys)
            return booster.fit_counts(data[0], nrounds)
        finally:
            if booster != NULL:
                del booster
            if data != NULL:
                del data
            if mat != NULL:
                del mat
