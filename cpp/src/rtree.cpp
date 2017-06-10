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

    RTree::RTree(const DataSet& data, const std::vector<size_t>& active) :
        m_active(active), m_yhat(data.mean_y(m_active)), m_is_leaf(true) {

        if (active.empty()) {
            throw std::invalid_argument("Must have at least one active row");
        }
        if (data.variance_y(active) > 1e-6) {
            auto split = best_split(data);

            if (split.is_valid()) {
                m_is_leaf = false;
                m_split_col = split.col_idx();
                m_split_val = split.value();

                std::vector<size_t> active_left;
                std::vector<size_t> active_right;
                populate_active(
                    data, m_split_col, m_split_val, active_left, active_right);

                m_left_child = std::make_unique<RTree>(data, active_left);
                m_right_child = std::make_unique<RTree>(data, active_right);
            }
        }
    }

    RTree::RTree(const DataSet& data) :
        RTree(data, sequential_ints(data.nrows())) {
    }

    void
    RTree::populate_active(
            const DataSet& data,
            const size_t col,
            const float split_val,
            std::vector<size_t>& active_left,
            std::vector<size_t>& active_right)
    const {
        for (const auto & row : m_active) {
            const auto x = data.x_at(row, col);
            if (x <= split_val) {
                active_left.push_back(row);
            } else {
                active_right.push_back(row);
            }
        }
    }

    double
    RTree::calc_total_err(
            const DataSet& data,
            const size_t col,
            const float split,
            const float yhat_l,
            const float yhat_r)
    const {
        double err = 0;

        for (const auto & row : m_active) {
            auto x_j = data.x_at(row, col);
            auto y_j = data.y_at(row);
            auto yhat = x_j <= split ? yhat_l : yhat_r;
            err += pow((y_j - yhat), 2.0);
        }
        return err;
    }

    void
    RTree::predict(const DataSet& data, std::vector<float>& yhat)
    const {
        if (m_is_leaf) {
            for (const auto & row : m_active) {
                yhat[row] = m_yhat;
            }
        } else {
            if (!m_left_child) {
                throw std::logic_error("Cannot predict on null left child node");
            }
            if (!m_right_child) {
                throw std::logic_error("Cannot predict on null right child node");
            }

            m_left_child->predict(data, yhat);
            m_right_child->predict(data, yhat);
        }
    }

    std::vector<float>
    RTree::predict(const DataSet& data)
    const {
        const auto nan = std::numeric_limits<double>::quiet_NaN();
        std::vector<float> yhat(data.nrows(), nan);

        predict(data, yhat);
        return yhat;
    }

    std::pair<float, float>
    RTree::fit_children(
            const DataSet& data,
            const size_t col,
            const float split_val)
    const {
        std::vector<size_t> active_left;
        std::vector<size_t> active_right;
        populate_active(data, col, split_val, active_left, active_right);
        const float yhat_l = data.mean_y(active_left);
        const float yhat_r = data.mean_y(active_right);

        return std::make_pair(yhat_l, yhat_r);
    }

    SplitData
    RTree::best_split(const DataSet& data)
    const {
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
                const auto yhat = fit_children(data, col, split);
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
