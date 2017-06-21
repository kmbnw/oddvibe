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
#include "dataset.h"
#include "algorithm_x.h"
#include "math_x.h"

namespace oddvibe {
    void RTree::predict(
            const FloatMatrix& mat,
            SizeVec& filter,
            FloatVec& yhat) const {
        if (m_is_leaf) {
            for (const auto & row : filter) {
                yhat[row] = m_yhat;
            }
        } else {
            const auto pivot = m_split.partition_idx(mat, filter);
            SizeVec lsplit(filter.begin(), pivot);
            SizeVec rsplit(pivot, filter.end());

            m_left->predict(mat, lsplit, yhat);
            m_right->predict(mat, rsplit, yhat);
        }
    }

    FloatVec RTree::predict(const FloatMatrix& mat) const {
        const auto nrows = mat.nrows();
        FloatVec yhats(nrows, floatNaN);
        auto filter = sequential_ints(nrows);

        predict(mat, filter, yhats);

        return yhats;
    }

    void RTree::fit(
            const Dataset<FloatMatrix, FloatVec>& dataset,
            const SizeVec& filter) {

        SizeVec defensive_copy(filter);
        fit(dataset, defensive_copy);
    }

    void RTree::fit(
            const Dataset<FloatMatrix, FloatVec>& dataset,
            SizeVec& filter) {
        if (filter.empty()) {
            throw std::invalid_argument("Must have at least one entry in filter");
        }

        const auto yhat = mean(dataset.ys(), filter.begin(), filter.end());
        if (std::isnan(yhat)) {
            throw std::logic_error("Prediction cannot be NaN");
        }

        auto is_leaf = true;
        SplitPoint split;
        std::unique_ptr<RTree> left;
        std::unique_ptr<RTree> right;

        if (variance(dataset.ys(), filter.begin(), filter.end()) > 1e-6) {
            split = best_split(dataset, filter);

            if (split.is_valid()) {
                is_leaf = false;

                const auto pivot = split.partition_idx(dataset.xs(), filter);
                SizeVec lsplit(filter.begin(), pivot);
                SizeVec rsplit(pivot, filter.end());

                left = std::make_unique<RTree>();
                right = std::make_unique<RTree>();

                left->fit(dataset, lsplit);
                right->fit(dataset, rsplit);
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

    SplitPoint RTree::best_split(
            const Dataset<FloatMatrix, FloatVec>& dataset,
            const SizeVec& filter) const {
        SplitPoint best;

        // TODO min size guard
        if (filter.empty()) {
            return best;
        }
        double best_err = doubleNaN;

        const auto ncols = dataset.ncols();
        for (size_t split_col = 0; split_col != ncols; ++split_col) {
            auto uniques = unique_x(dataset.xs(), split_col, filter);

            if (uniques.size() < 2) {
                continue;
            }

            for (const auto & split_val : uniques) {
                SplitPoint split(split_val, split_col);

                // total squared error for left and right side of split_val
                const auto err = dataset.calc_total_err(split, filter);

                // TODO randomly allow the same error as best to 'win'
                if (!std::isnan(err) && (std::isnan(best_err) || err < best_err)) {
                    best = std::move(split);
                    best_err = err;
                }
            }
        }
        return best;
    }
}
