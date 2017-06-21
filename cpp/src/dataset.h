/*
 * Copyright 2016-2017 Krysta M Bouzek
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <vector>
#include <unordered_set>
#include <utility>
#include "math_x.h"

#ifndef KMBNW_TRAIN_DATA_H
#define KMBNW_TRAIN_DATA_H

namespace oddvibe {
    // follow R's NumericVector API where necessary to facilitate easier use
    class FloatMatrix {
        public:
            /**
             * Create new instance.
             * @param[in] ncols: Number of columns/features.
             * @param[in] xs: Flattened matrix of features: row0 followed by
             * row1, etc.
             */
            FloatMatrix(const size_t ncols, FloatVec&& xs);

            FloatMatrix(FloatMatrix&& other) = default;
            FloatMatrix(const FloatMatrix& other) = default;
            FloatMatrix& operator=(const FloatMatrix& other) = default;
            FloatMatrix& operator=(FloatMatrix&& other) = default;
            ~FloatMatrix() = default;

            float operator() (const size_t row, const size_t col) const;

            /**
             * Number of rows.
             */
            size_t nrows() const;

            /**
             * Number of columns (features).
             */
            size_t ncols() const;

            private:
                size_t m_nrows = 0;
                size_t m_ncols = 0;
                FloatVec m_xs;

                size_t x_index(const size_t row, const size_t col) const;

    };
}
#endif //KMBNW_TRAIN_DATA_H
