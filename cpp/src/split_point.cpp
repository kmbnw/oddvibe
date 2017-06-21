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

    // total squared error for left and right side of split_val
    double SplitPoint::calc_total_err(
            const FloatMatrix& mat,
            const FloatVec& ys,
            const SizeConstIter first,
            const SizeConstIter last) const {

        if (first == last) {
            return doubleNaN;
        }

        // TODO discard this temporary
        SizeVec part(first, last);
        const auto pivot = partition_idx(mat, part);

        const auto avg = partitioned_mean(ys, part.begin(), pivot, part.end());
        const float yhat_l = avg.first;
        const float yhat_r = avg.second;

        if (std::isnan(yhat_l) || std::isnan(yhat_r)) {
            return doubleNaN;
        }


        double err = 0;
        for (auto row = first; row != last; row = std::next(row)) {
            const auto x_j = mat(*row, m_split_col);
            const auto y_j = ys[*row];
            const auto yhat_j = x_j <= m_split_val ? yhat_l : yhat_r;
            err += pow((y_j - yhat_j), 2.0);
        }
        return err;
    }
}
