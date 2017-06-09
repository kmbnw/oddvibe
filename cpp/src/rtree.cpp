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

#include "rtree.h"
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace oddvibe {
    RTree::RTree(const DataSet& data, const std::vector<bool>& active) :
        m_active(active) {

        if (std::find(active.begin(), active.end(), true) != active.end()) {
            throw std::invalid_argument("Must have at least one active row");
        }
        if (data.variance_y(active) > 1e-6) {
            auto split = best_split(data);

            if (split.is_valid()) {
                const auto split_col = split.col_idx();
                const auto split_val = split.value();
                const auto nrows = data.nrows();
                std::vector<bool> left_filter(nrows, false);
                std::vector<bool> right_filter(nrows, false);
                populate_filter(data, split_col, split_val, left_filter, right_filter);

                m_left_child = std::make_unique<RTree>(data, left_filter);
                m_right_child = std::make_unique<RTree>(data, right_filter);
            }
        }
    }

    RTree::RTree(const DataSet& data) :
        RTree(data, std::vector<bool>(true, data.nrows())) {
    }

    void RTree::populate_filter(
            const DataSet& data,
            const size_t col,
            const float split_val,
            std::vector<bool>& left_filter,
            std::vector<bool>& right_filter) const {
        const auto nrows = data.nrows();

        for (size_t row = 0; row != nrows; ++row) {
            if (m_active[row]) {
                const auto x = data.x_at(row, col);
                if (x <= split_val) {
                    left_filter[row] = true;
                } else {
                    right_filter[row] = true;
                }
            }
        }
    }

    double RTree::calc_total_err(
            const DataSet& data,
            const size_t col,
            const float split,
            const float yhat_l,
            const float yhat_r) const {
        double err = 0;
        const auto nrows = data.nrows();

        for (size_t row_j = 0; row_j != nrows; ++row_j) {
            if (m_active[row_j]) {
                auto x_j = data.x_at(row_j, col);
                auto y_j = data.y_at(row_j);
                auto yhat = x_j <= split ? yhat_l : yhat_r;
                err += pow((y_j - yhat), 2.0);
            }
        }
        return err;
    }

    float RTree::predict(const DataSet& data) const {
        return data.mean_y(m_active);
    }

    std::pair<float, float>
    RTree::predict_split(
            const DataSet& data,
            const size_t col,
            const float split_val) const {
        const auto nrows = data.nrows();

        std::vector<bool> left_filter(nrows, false);
        std::vector<bool> right_filter(nrows, false);
        populate_filter(data, col, split_val, left_filter, right_filter);
        const float yhat_l = data.mean_y(left_filter);
        const float yhat_r = data.mean_y(right_filter);

        return std::make_pair(yhat_l, yhat_r);
    }

    SplitData
    RTree::best_split(const DataSet& data) const {
        double best_err = std::numeric_limits<double>::quiet_NaN();
        size_t best_feature = -1;
        float best_value = std::numeric_limits<float>::quiet_NaN();
        bool init = false;

        const auto ncols = data.ncols();
        for (size_t col = 0; col != ncols; ++col) {
            auto uniques = data.unique_x(col, m_active);

            if (uniques.size() < 2) {
                continue;
            }

            for (const auto & split : uniques) {
                // calculate yhat for left and right side of split
                const auto yhat = predict_split(data, col, split);
                const auto yhat_l = yhat.first;
                const auto yhat_r = yhat.second;

                if (std::isnan(yhat_l) || std::isnan(yhat_r)) {
                    continue;
                }

                // total squared error for left and right side of split
                const auto err = calc_total_err(data, col, split, yhat_l, yhat_r);

                // TODO randomly allow the same error as best to 'win'
                if (!init || (!std::isnan(err) && err < best_err)) {
                    init = true;
                    best_err = err;
                    best_value = split;
                    best_feature = col;
                }
            }
        }
        return SplitData(best_value, best_feature, best_err);
    }
}
