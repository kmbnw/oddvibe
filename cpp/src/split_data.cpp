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
#include "split_data.h"

namespace oddvibe {
    SplitData::SplitData(const float split_val, const size_t split_col):
        m_split_val(split_val),
        m_split_col(split_col) {
    }

    bool SplitData::is_valid() const {
        return !std::isnan(m_split_val);
    }

    float SplitData::split_val() const {
        return m_split_val;
    }

    size_t SplitData::split_col() const {
        return m_split_col;
    }

    void
    SplitData::fill_row_idx(
            const DataSet& data,
            const SizeVec& filter,
            SizeVec& left_rows,
            SizeVec& right_rows)
    const {
        for (const auto & row : filter) {
            const auto x = data.x_at(row, m_split_col);
            if (x <= m_split_val) {
                left_rows.push_back(row);
            } else {
                right_rows.push_back(row);
            }
        }
    }

    // total squared error for left and right side of split_val
    double
    SplitData::calc_total_err(const DataSet& data, const SizeVec& filter) const {
        SizeVec left_idx;
        SizeVec right_idx;
        fill_row_idx(data, filter, left_idx, right_idx);

        const float yhat_l = data.mean_y(left_idx);
        if (std::isnan(yhat_l)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const float yhat_r = data.mean_y(right_idx);
        if (std::isnan(yhat_r)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double err = 0;
        for (const auto & row : filter) {
            const auto x_j = data.x_at(row, m_split_col);
            const auto y_j = data.y_at(row);
            const auto yhat_j = x_j <= m_split_val ? yhat_l : yhat_r;
            err += pow((y_j - yhat_j), 2.0);
        }
        return err;
    }
}
