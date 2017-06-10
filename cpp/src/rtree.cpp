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
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace oddvibe {
    std::vector<size_t>
    sequential_ints(const size_t len) {
        std::vector<size_t> seq(len, 0);
        std::iota(seq.begin(), seq.end(), 0);
        return seq;
    }

    RTree::RTree(const DataSet& data, const std::vector<size_t>& rows) :
        m_yhat(data.mean_y(rows)), m_is_leaf(true) {

        if (std::isnan(m_yhat)) {
            throw std::logic_error("Prediction cannot be NaN");
        }
        if (rows.empty()) {
            throw std::invalid_argument("Must have at least one active row");
        }
        if (data.variance_y(rows) > 1e-6) {
            auto split = best_split(data, rows);

            if (split.is_valid()) {
                m_is_leaf = false;
                m_split_col = split.col_idx();
                m_split_val = split.value();

                std::vector<size_t> left_idx;
                std::vector<size_t> right_idx;
                fill_row_idx(
                    data, m_split_col, m_split_val, rows, left_idx, right_idx);

                m_left_child = std::make_unique<RTree>(data, left_idx);
                m_right_child = std::make_unique<RTree>(data, right_idx);
            }
        }

        if (!m_is_leaf) {
            if (!m_left_child) {
                throw std::logic_error("Cannot have a null left child node");
            }
            if (!m_right_child) {
                throw std::logic_error("Cannot have a null right child node");
            }
        }
    }

    RTree::RTree(const DataSet& data) :
        RTree(data, sequential_ints(data.nrows())) {
    }

    void
    RTree::fill_row_idx(
            const DataSet& data,
            const size_t col,
            const float split_val,
            const std::vector<size_t>& rows,
            std::vector<size_t>& left_rows,
            std::vector<size_t>& right_rows)
    const {
        for (const auto & row : rows) {
            const auto x = data.x_at(row, col);
            if (x <= split_val) {
                left_rows.push_back(row);
            } else {
                right_rows.push_back(row);
            }
        }
    }

    double
    RTree::calc_total_err(
            const DataSet& data,
            const std::vector<size_t>& rows,
            const size_t col,
            const float split,
            const float yhat_l,
            const float yhat_r)
    const {
        double err = 0;

        for (const auto & row : rows) {
            auto x_j = data.x_at(row, col);
            auto y_j = data.y_at(row);
            auto yhat = x_j <= split ? yhat_l : yhat_r;
            err += pow((y_j - yhat), 2.0);
        }
        return err;
    }

    void
    RTree::fill_active(
            const DataSet& data,
            const std::vector<bool>& init_active,
            std::vector<bool>& l_active,
            std::vector<bool>& r_active)
    const {
        const auto nrows = data.nrows();
        for (size_t row = 0; row != nrows; ++row) {
            if (init_active[row]) {
                auto x_j = data.x_at(row, m_split_col);
                if (x_j <= m_split_val) {
                    l_active[row] = true;
                } else {
                    r_active[row] = true;
                }
            }
        }
    }

    void
    RTree::predict(
            const DataSet& data,
            const std::vector<bool>& active,
            std::vector<float>& yhat)
    const {
        const auto nrows = data.nrows();
        if (m_is_leaf) {
            for (size_t row = 0; row != nrows; ++row) {
                if (active[row]) {
                    yhat[row] = m_yhat;
                }
            }
        } else {
            std::vector<bool> l_active(nrows, false);
            std::vector<bool> r_active(nrows, false);
            fill_active(data, active, l_active, r_active);

            m_left_child->predict(data, l_active, yhat);
            m_right_child->predict(data, r_active, yhat);
        }
    }

    std::vector<float>
    RTree::predict(const DataSet& data)
    const {
        const auto nan = std::numeric_limits<double>::quiet_NaN();
        const auto nrows = data.nrows();
        std::vector<float> yhats(nrows, nan);
        std::vector<bool> active(nrows, true);

        predict(data, active, yhats);

        const auto pred = [](float yhat) { return std::isnan(yhat); };
        if (std::find_if(yhats.begin(), yhats.end(), pred) != yhats.end()) {
            throw std::logic_error("One or more predictions were NaN");
        }

        return yhats;
    }

    std::pair<float, float>
    RTree::fit_children(
            const DataSet& data,
            const std::vector<size_t>& rows,
            const size_t col,
            const float split_val)
    const {
        std::vector<size_t> left_idx;
        std::vector<size_t> right_idx;
        fill_row_idx(data, col, split_val, rows, left_idx, right_idx);
        const float yhat_l = data.mean_y(left_idx);
        const float yhat_r = data.mean_y(right_idx);

        return std::make_pair(yhat_l, yhat_r);
    }

    SplitData
    RTree::best_split(const DataSet& data, const std::vector<size_t>& rows)
    const {
        double best_err = std::numeric_limits<double>::quiet_NaN();
        size_t best_feature = -1;
        float best_value = std::numeric_limits<float>::quiet_NaN();
        bool init = false;

        const auto ncols = data.ncols();
        for (size_t col = 0; col != ncols; ++col) {
            auto uniques = data.unique_x(col, rows);

            if (uniques.size() < 2) {
                continue;
            }

            for (const auto & split : uniques) {
                // calculate yhat for left and right side of split
                const auto yhat = fit_children(data, rows, col, split);
                const auto yhat_l = yhat.first;
                const auto yhat_r = yhat.second;

                if (std::isnan(yhat_l) || std::isnan(yhat_r)) {
                    continue;
                }

                // total squared error for left and right side of split
                const auto err = calc_total_err(data, rows, col, split, yhat_l, yhat_r);

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
