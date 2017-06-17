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

#include <limits>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "train_data.h"

namespace oddvibe {
     DataSet::DataSet(const size_t ncols, const FloatVec& xs, const FloatVec& ys):
            m_nrows(ys.size()), m_ncols(ncols) {
        if (ys.size() != xs.size() / ncols) {
            throw std::invalid_argument("xs and ys must have same number of rows");
        }
        m_xs = xs;
        m_ys = ys;
    }

    float
    DataSet::y_at(const size_t row)
    const {
        if (row < m_nrows) {
            return m_ys[row];
        }
        throw std::out_of_range("row out of range");
    }

    float
    DataSet::x_at(const size_t row, const size_t col)
    const {
        if (row >= m_nrows) {
            throw std::out_of_range("row out of range");
        }
        if (col >= m_ncols) {
            throw std::out_of_range("col out of range");
        }

        return m_xs[x_index(row, col)];
    }

    size_t DataSet::nrows() const {
        return m_nrows;
    }

    size_t DataSet::ncols() const {
        return m_ncols;
    }

    size_t
    DataSet::x_index(const size_t row, const size_t col)
    const {
        return (row * m_ncols) + col;
    }

    std::unordered_set<float>
    DataSet::unique_x(const size_t col, const SizeVec& row_idx) const {
        std::unordered_set<float> uniques;

        for (const auto & row : row_idx) {
            uniques.insert(m_xs[x_index(row, col)]);
        }
        return uniques;
    }

    double
    DataSet::mean_y(const SizeVec& row_idx) const {
        if (m_ys.empty()) {
            return 0;
        }

        size_t count = 0;
        double total = 0;

        for (const auto & row : row_idx) {
            total += m_ys[row];
            ++count;
        }
        return (count < 1 ? 0 : total / count);
    }

    double
    DataSet::variance_y(const SizeVec& row_idx) const {
        if (m_ys.empty()) {
            return 0;
        }

        size_t count = 0;
        double total = 0;
        const auto mean = mean_y(row_idx);

        for (const auto & row : row_idx) {
            total += pow(m_ys[row] - mean, 2);
            ++count;
        }

        const auto nan = std::numeric_limits<double>::quiet_NaN();
        return (count < 1 ? nan : total / count);
    }

    DoubleVec
    DataSet::loss(const FloatVec& yhats) const {
        if (yhats.size() != m_ys.size()) {
            throw std::logic_error("Observed and predicted must be same size");
        }
        DoubleVec loss(yhats.size(), 0);
        std::transform(
            yhats.begin(),
            yhats.end(),
            m_ys.begin(),
            loss.begin(),
            rmse_loss);
        return loss;
    }

    std::pair<BoolVec, BoolVec>
    DataSet::partition_rows(const SplitPoint& split, const BoolVec& filter) const {
        const auto split_col = split.split_col();
        const auto split_val = split.split_val();
        BoolVec left(m_nrows, false);
        BoolVec right(m_nrows, false);
        for (size_t row = 0; row != m_nrows; ++row) {
            if (filter[row]) {
                const auto x_j = m_xs[x_index(row, split_col)];
                if (x_j <= split_val) {
                    left[row] = true;
                } else {
                    right[row] = true;
                }
            }
        }
        return std::make_pair(std::move(left), std::move(right));
    }

    std::pair<SizeVec, SizeVec>
    DataSet::partition_rows(const SplitPoint& split, const SizeVec& filter) const {
        const auto split_col = split.split_col();
        const auto split_val = split.split_val();
        SizeVec left;
        SizeVec right;

        for (const auto & row : filter) {
            const auto x_j = m_xs[x_index(row, split_col)];
            if (x_j <= split_val) {
                left.push_back(row);
            } else {
                right.push_back(row);
            }
        }
        return std::make_pair(std::move(left), std::move(right));
    }

    // total squared error for left and right side of split_val
    double
    DataSet::calc_total_err(const SplitPoint& split, const SizeVec& filter) const {
        const auto part = partition_rows(split, filter);

        const float yhat_l = mean_y(part.first);
        if (std::isnan(yhat_l)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const float yhat_r = mean_y(part.second);
        if (std::isnan(yhat_r)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const auto split_col = split.split_col();
        const auto split_val = split.split_val();

        double err = 0;
        for (const auto & row : filter) {
            const auto x_j = m_xs[x_index(row, split_col)];
            const auto y_j = m_ys[row];
            const auto yhat_j = x_j <= split_val ? yhat_l : yhat_r;
            err += pow((y_j - yhat_j), 2.0);
        }
        return err;
    }
}
