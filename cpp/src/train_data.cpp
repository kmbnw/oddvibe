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

        return m_xs[x_index(row_idx, col_idx)];
    }

    size_t DataSet::nrows() const {
        return m_nrows;
    }

    size_t DataSet::ncols() const {
        return m_ncols;
    }

    size_t DataSet::x_index(const size_t row, const size_t col) const {
        return (row * m_ncols) + col;
    }

    std::unordered_set<float>
    DataSet::unique_x(const size_t col, const std::vector<size_t>& active) const {
        std::unordered_set<float> uniques;

        for (const auto & row : active) {
            uniques.insert(m_xs[x_index(row, col)]);
        }
        return uniques;
    }


    double DataSet::mean_y(const std::vector<size_t>& active) const {
        if (m_ys.empty()) {
            return 0;
        }

        size_t count = 0;
        double total = 0;

        for (const auto & row : active) {
            total += m_ys[row];
            ++count;
        }
        return (count < 1 ? 0 : total / count);
    }

    double DataSet::variance_y(const std::vector<size_t>& active) const {
        if (m_ys.empty()) {
            return 0;
        }

        size_t count = 0;
        double total = 0;
        const auto mean = mean_y(active);

        for (const auto & row : active) {
            total += pow(m_ys[row] - mean, 2);
            ++count;
        }

        const auto nan = std::numeric_limits<double>::quiet_NaN();
        return (count < 1 ? nan : total / count);
    }
}
