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
#ifndef KMBNW_ODVB_SPLITPOINT_H
#define KMBNW_ODVB_SPLITPOINT_H

#include <cstddef>
#include <limits>
#include <algorithm>
#include <utility>
#include "defs_x.h"
#include "math_x.h"

namespace oddvibe {
    class SplitPoint {
        public:
            SplitPoint() = default;

            SplitPoint(const float split_val, const size_t split_col);

            SplitPoint(SplitPoint&& other) = default;
            SplitPoint(const SplitPoint& other) = default;
            SplitPoint& operator=(const SplitPoint& other) = default;
            SplitPoint& operator=(SplitPoint&& other) = default;
            ~SplitPoint() = default;

            float split_val() const;

            size_t split_col() const;

            bool is_valid() const;


            template<typename MatrixT, typename IndexSeqT>
            typename IndexSeqT::iterator
            partition_idx(const MatrixT& mat, IndexSeqT& filter) const {
                return std::partition(
                    filter.begin(),
                    filter.end(),
                    [self = this, &mat](const auto & row){
                        return mat(row, self->m_split_col) <= self->m_split_val;
                    });
            }

            // total squared error for left and right side of split_val
            template<typename MatrixT, typename VectorT>
            double calc_total_err(
                    const MatrixT& xs,
                    const VectorT& ys,
                    const SizeVec& filter) const {
                if (filter.empty()) {
                    return doubleNaN;
                }

                const auto avg = partitioned_mean(xs, ys, filter);
                const float yhat_l = avg.first;
                const float yhat_r = avg.second;

                if (std::isnan(yhat_l) || std::isnan(yhat_r)) {
                    return doubleNaN;
                }

                double err = 0;
                for (const auto & row : filter) {
                    const auto x_j = xs(row, m_split_col);
                    const auto y_j = ys[row];
                    const auto yhat_j = x_j <= m_split_val ? yhat_l : yhat_r;
                    err += pow((y_j - yhat_j), 2.0);
                }
                return err;
            }

        private:
            float m_split_val = floatNaN;
            size_t m_split_col = 0;

            template<typename MatrixT, typename VectorT>
            std::pair<float, float>
            partitioned_mean(
                    const MatrixT& xs,
                    const VectorT& ys,
                    const SizeVec& rows) const {
                SizeVec part(rows);
                const auto pivot = partition_idx(xs, part);

                auto first_p = part.begin();
                auto last_p = part.end();
                // std::optional would be good here, but not at C++ 17 yet
                // for this install
                if (first_p == last_p || pivot == first_p || pivot == last_p) {
                    return std::make_pair(floatNaN, floatNaN);
                }

                const float yhat_l = mean(ys, first_p, pivot);
                if (std::isnan(yhat_l)) {
                    return std::make_pair(floatNaN, floatNaN);
                }

                const float yhat_r = mean(ys, pivot, last_p);
                if (std::isnan(yhat_r)) {
                    return std::make_pair(floatNaN, floatNaN);
                }
                return std::make_pair(yhat_l, yhat_r);
            }
    };

    template<typename MatrixT, typename VectorT>
    SplitPoint
    best_split(const MatrixT& xs, const VectorT& ys, const SizeVec& filter) {
        SplitPoint best;

        // TODO min size guard
        if (filter.empty()) {
            return best;
        }
        double best_err = doubleNaN;

        const auto ncols = xs.ncols();
        for (size_t split_col = 0; split_col != ncols; ++split_col) {
            auto uniques = unique_x(xs, split_col, filter);

            if (uniques.size() < 2) {
                continue;
            }

            for (const auto & split_val : uniques) {
                SplitPoint split(split_val, split_col);

                // total squared error for left and right side of split_val
                const auto err = split.calc_total_err(xs, ys, filter);

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
#endif //KMBNW_ODVB_SPLITPOINT_H
