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

#include <utility>
#include <limits>
#include <stdexcept>
#include <cmath>
#include "train_data.h"

namespace oddvibe {
     DataSet::DataSet(
            const size_t ncols,
            const std::vector<float>& xs,
            const std::vector<float>& ys):
            m_nrows(ys.size()), m_ncols(ncols), m_xs(xs), m_ys(ys) {
        if (ys.size() != xs.size() / ncols) {
            throw std::invalid_argument("xs and ys must have same number of rows");
        }
    }

    float DataSet::y_at(const size_t row_idx) const {
        if (row_idx < m_nrows) {
            return m_ys[row_idx];
        }
        throw std::out_of_range("row_idx out of range");
    }

    float DataSet::x_at(const size_t row_idx, const size_t col_idx) const {
        if (row_idx >= m_nrows) {
            throw std::out_of_range("row_idx out of range");
        }
        if (col_idx >= m_ncols) {
            throw std::out_of_range("col_idx out of range");
        }

        return m_xs[(row_idx * m_ncols) + col_idx];
    }

    size_t DataSet::nrows() const {
        return m_nrows;
    }

    size_t DataSet::ncols() const {
        return m_ncols;
    }

    float DataSet::filtered_mean(
            const std::vector<bool> &row_filter) const {
        double sum = 0;
        size_t count = 0;
        for (size_t i = 0; i != m_ys.size(); ++i) {
            if (row_filter[i]) {
                sum += m_ys[i];
                ++count;
            }
        }

        return (float) (sum / count);
    }
}
