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
#include "algorithm_x.h"

namespace oddvibe {
    void RTree::predict(
            const FloatMatrix& mat,
            const SizeConstIter first,
            const SizeConstIter last,
            FloatVec& yhat) const {
        if (m_is_leaf) {
            for (auto row = first; row != last; row = std::next(row)) {
                yhat[*row] = m_yhat;
            }
        } else {
            SizeVec part(first, last);
            const auto pivot = m_split.partition_idx(mat, part);

            m_left->predict(mat, part.begin(), pivot, yhat);
            m_right->predict(mat, pivot, part.end(), yhat);
        }
    }

    FloatVec RTree::predict(const FloatMatrix& mat) const {
        const auto nrows = mat.nrows();
        FloatVec yhats(nrows, floatNaN);
        auto filter = sequential_ints(nrows);

        predict(mat, filter.begin(), filter.end(), yhats);

        return yhats;
    }

    void RTree::fit(
            const FloatMatrix& mat,
            const FloatVec& ys,
            const SizeVec& filter) {
        fit(mat, ys, filter.begin(), filter.end());
    }

    void RTree::fit(
            const FloatMatrix& mat,
            const FloatVec& ys,
            const SizeConstIter first,
            const SizeConstIter last) {
        if (first == last) {
            throw std::invalid_argument("Must have at least one entry in filter");
        }

        const auto yhat = mean(ys, first, last);
        if (std::isnan(yhat)) {
            throw std::logic_error("Prediction cannot be NaN");
        }

        auto is_leaf = true;
        SplitPoint split;
        std::unique_ptr<RTree> left;
        std::unique_ptr<RTree> right;

        if (variance(ys, first, last) > 1e-6) {
            split = best_split(mat, ys, first, last);

            if (split.is_valid()) {
                is_leaf = false;

                SizeVec part(first, last);
                const auto pivot = split.partition_idx(mat, part);

                left = std::make_unique<RTree>();
                right = std::make_unique<RTree>();

                left->fit(mat, ys, part.begin(), pivot);
                right->fit(mat, ys, pivot, part.end());
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
            const FloatMatrix& mat,
            const FloatVec& ys,
            const SizeConstIter first,
            const SizeConstIter last) const {
        SplitPoint best;

        if (first == last) {
            return best;
        }

        double best_err = doubleNaN;
        bool init = false;

        const auto ncols = mat.ncols();
        for (size_t split_col = 0; split_col != ncols; ++split_col) {
            auto uniques = mat.unique_x(split_col, first, last);

            if (uniques.size() < 2) {
                continue;
            }

            for (const auto & split_val : uniques) {
                SplitPoint split(split_val, split_col);

                // total squared error for left and right side of split_val
                const auto err = split.calc_total_err(mat, ys, first, last);

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
