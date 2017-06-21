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

#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "dataset.h"

namespace oddvibe {
    FloatMatrix::FloatMatrix(const size_t ncols, FloatVec&& xs) {
        if (xs.empty()) {
            if (ncols > 0) {
                throw std::invalid_argument("Cannot set ncols for empty vector");
            }
        } else {
            if (xs.size() % ncols != 0) {
                throw std::invalid_argument("Invalid shape for input vector");
            }
            m_ncols = ncols;
            m_nrows = xs.size() / ncols;
            m_xs = std::move(xs);
        }
    }

    float FloatMatrix::operator() (const size_t row, const size_t col) const {
        return m_xs[x_index(row, col)];
    }

    size_t FloatMatrix::nrows() const {
        return m_nrows;
    }

    size_t FloatMatrix::ncols() const {
        return m_ncols;
    }

    size_t FloatMatrix::x_index(const size_t row, const size_t col) const {
        return (row * m_ncols) + col;
    }
}
