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
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stack>
#include "rtree.h"

namespace oddvibe {
    void
    RTree::predict(
            const DataSet& data,
            const BoolVec& filter,
            FloatVec& yhat)
    const {
        const auto nrows = data.nrows();
        if (m_is_leaf) {
            for (size_t row = 0; row != nrows; ++row) {
                if (filter[row]) {
                    yhat[row] = m_yhat;
                }
            }
        } else {
            BoolVec left_filter(nrows, false);
            BoolVec right_filter(nrows, false);
            fill_filter(data, filter, left_filter, right_filter);

            m_left->predict(data, left_filter, yhat);
            m_right->predict(data, right_filter, yhat);
        }
    }

    FloatVec RTree::predict(const DataSet& data) const {
        const auto nan = std::numeric_limits<double>::quiet_NaN();
        const auto nrows = data.nrows();
        FloatVec yhats(nrows, nan);
        BoolVec filter(nrows, true);

        predict(data, filter, yhats);

        return yhats;
    }

    void
    RTree::fill_filter(
            const DataSet& data,
            const BoolVec& init_filter,
            BoolVec& left_filter,
            BoolVec& right_filter)
    const {
        const auto nrows = data.nrows();
        const auto split_col = m_split.split_col();
        const auto split_val = m_split.split_val();

        for (size_t row = 0; row != nrows; ++row) {
            if (init_filter[row]) {
                auto x_j = data.x_at(row, split_col);
                if (x_j <= split_val) {
                    left_filter[row] = true;
                } else {
                    right_filter[row] = true;
                }
            }
        }
    }

    void
    RTree::fit(const DataSet& data, const SizeVec& filter) {
        if (filter.empty()) {
            throw std::invalid_argument("Must have at least one entry in filter");
        }

        const auto yhat = data.mean_y(filter);
        auto is_leaf = true;
        SplitData split;
        std::unique_ptr<RTree> left;
        std::unique_ptr<RTree> right;

        if (std::isnan(yhat)) {
            throw std::logic_error("Prediction cannot be NaN");
        }

        if (data.variance_y(filter) > 1e-6) {
            split = best_split(data, filter);

            if (split.is_valid()) {
                is_leaf = false;

                SizeVec left_filter;
                SizeVec right_filter;
                split.fill_row_idx(data, filter, left_filter, right_filter);

                left = std::make_unique<RTree>();
                right = std::make_unique<RTree>();

                left->fit(data, left_filter);
                right->fit(data, right_filter);
            }
        }

        if (!is_leaf) {
            if (!left) {
                throw std::logic_error("Cannot have a null left child node");
            }
            if (!right) {
                throw std::logic_error("Cannot have a null right child node");
            }

            m_split = split;
            m_left = std::move(left);
            m_right = std::move(right);
        }

        m_yhat = yhat;
        m_is_leaf = is_leaf;
    }

    SplitData
    RTree::best_split(const DataSet& data, const SizeVec& filter)
    const {
        SplitData best;
        double best_err = std::numeric_limits<double>::quiet_NaN();
        bool init = false;

        const auto ncols = data.ncols();
        for (size_t split_col = 0; split_col != ncols; ++split_col) {
            auto uniques = data.unique_x(split_col, filter);

            if (uniques.size() < 2) {
                continue;
            }

            for (const auto & split_val : uniques) {
                SplitData split(split_val, split_col);

                // total squared error for left and right side of split_val
                const auto err = split.calc_total_err(data, filter);

                // TODO randomly allow the same error as best to 'win'
                if (!init || (!std::isnan(err) && err < best_err)) {
                    init = true;
                    best = split;
                    best_err = err;
                }
            }
        }
        return best;
    }
}
