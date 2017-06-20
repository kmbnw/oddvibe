/*
 * Copyright 2017 Krysta M Bouzek
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

#include <cmath>
#include <algorithm>
#include "split_point.h"

namespace oddvibe {
    SplitPoint::SplitPoint(const float split_val, const size_t split_col):
        m_split_val(split_val),
        m_split_col(split_col) {
    }

    bool SplitPoint::is_valid() const {
        return !std::isnan(m_split_val);
    }

    float SplitPoint::split_val() const {
        return m_split_val;
    }

    size_t SplitPoint::split_col() const {
        return m_split_col;
    }

    std::pair<BoolVec, BoolVec>
    SplitPoint::partition_rows(const FloatMatrix& mat, const BoolVec& filter)
    const {
        const auto nrows = mat.nrows();
        BoolVec left(nrows, false);
        BoolVec right(nrows, false);
        for (size_t row = 0; row != nrows; ++row) {
            if (filter[row]) {
                const auto x_j = mat(row, m_split_col);
                if (x_j <= m_split_val) {
                    left[row] = true;
                } else {
                    right[row] = true;
                }
            }
        }
        return std::make_pair(std::move(left), std::move(right));
    }

    std::pair<SizeVec, SizeVec>
    SplitPoint::partition_rows(const FloatMatrix& mat, const SizeVec& filter)
    const {
        SizeVec left;
        SizeVec right;

        for (const auto & row : filter) {
            const auto x_j = mat(row, m_split_col);
            if (x_j <= m_split_val) {
                left.push_back(row);
            } else {
                right.push_back(row);
            }
        }
        return std::make_pair(std::move(left), std::move(right));
    }

    SizeIter
    SplitPoint::partition_idx(const FloatMatrix& mat, SizeVec& filter) const {
        return std::partition(
            filter.begin(),
            filter.end(),
            [col = m_split_col, val = m_split_val, &mat](const auto & row){
                return mat(row, col) <= val;
            });
    }

    // total squared error for left and right side of split_val
    double SplitPoint::calc_total_err(
            const FloatMatrix& mat,
            const FloatVec& ys,
            const SizeVec& filter) const {
        const auto part = partition_rows(mat, filter);

        const float yhat_l = mean(ys, part.first);
        if (std::isnan(yhat_l)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const float yhat_r = mean(ys, part.second);
        if (std::isnan(yhat_r)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double err = 0;
        for (const auto & row : filter) {
            const auto x_j = mat(row, m_split_col);
            const auto y_j = ys[row];
            const auto yhat_j = x_j <= m_split_val ? yhat_l : yhat_r;
            err += pow((y_j - yhat_j), 2.0);
        }
        return err;
    }
}
