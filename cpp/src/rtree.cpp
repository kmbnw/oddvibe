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
            const auto part = data.partition_rows(m_split, filter);

            m_left->predict(data, part.first, yhat);
            m_right->predict(data, part.second, yhat);
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
    RTree::fit(const DataSet& data, const SizeVec& filter) {
        if (filter.empty()) {
            throw std::invalid_argument("Must have at least one entry in filter");
        }

        const auto yhat = data.mean_y(filter);
        auto is_leaf = true;
        SplitPoint split;
        std::unique_ptr<RTree> left;
        std::unique_ptr<RTree> right;

        if (std::isnan(yhat)) {
            throw std::logic_error("Prediction cannot be NaN");
        }

        if (data.variance_y(filter) > 1e-6) {
            split = best_split(data, filter);

            if (split.is_valid()) {
                is_leaf = false;

                const auto part = data.partition_rows(split, filter);

                left = std::make_unique<RTree>();
                right = std::make_unique<RTree>();

                left->fit(data, part.first);
                right->fit(data, part.second);
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

    SplitPoint
    RTree::best_split(const DataSet& data, const SizeVec& filter)
    const {
        SplitPoint best;
        double best_err = std::numeric_limits<double>::quiet_NaN();
        bool init = false;

        const auto ncols = data.ncols();
        for (size_t split_col = 0; split_col != ncols; ++split_col) {
            auto uniques = data.unique_x(split_col, filter);

            if (uniques.size() < 2) {
                continue;
            }

            for (const auto & split_val : uniques) {
                SplitPoint split(split_val, split_col);

                // total squared error for left and right side of split_val
                const auto err = data.calc_total_err(split, filter);

                // TODO randomly allow the same error as best to 'win'
                if (!init || (!std::isnan(err) && err < best_err)) {
                    init = true;
                    best = std::move(split);
                    best_err = err;
                }
            }
        }
        return best;
    }
}
