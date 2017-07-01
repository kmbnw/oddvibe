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
#include <future>
#include <unordered_set>
#include "defs_x.h"
#include "math_x.h"
#include "dataset.h"

namespace oddvibe {
    class SplitPoint {
        public:
            SplitPoint() = default;

            SplitPoint(const size_t split_col, const float split_val);

            SplitPoint(SplitPoint&& other) = default;
            SplitPoint(const SplitPoint& other) = default;
            SplitPoint& operator=(const SplitPoint& other) = default;
            SplitPoint& operator=(SplitPoint&& other) = default;
            ~SplitPoint() = default;

            float split_val() const;

            size_t split_col() const;

            bool is_valid() const;


            template <typename MatrixT, typename IndexSeqT>
            typename IndexSeqT::iterator
            partition_idx(const MatrixT& mat, IndexSeqT& filter) const {
                return std::partition(
                    filter.begin(),
                    filter.end(),
                    [this, &mat](const size_t row){
                        return mat(row, this->m_split_col) <= this->m_split_val;
                    });
            }

        private:
            size_t m_split_col = 0;
            float m_split_val = floatNaN;
    };

    // total squared error for left and right side of split_val
    template <typename MatrixT, typename VectorT>
    double calc_total_err(
            const size_t split_col,
            const float split_val,
            const Dataset<MatrixT, VectorT>& data,
            const SizeVec& filter) {
        size_t count_l = 0;
        size_t count_r = 0;
        double yhat_l = 0;
        double yhat_r = 0;

        const MatrixT& xs = data.xs();
        const VectorT& ys = data.ys();
        for (const auto & row : filter) {
            if (xs(row, split_col) <= split_val) {
                yhat_l += ys[row];
                ++count_l;
            } else {
                yhat_r += ys[row];
                ++count_r;
            }
        }

        if (count_l == 0 || count_r == 0) {
            return doubleMax;
        }
        yhat_l /= count_l;
        yhat_r /= count_r;

        double err = 0;
        for (const auto & row : filter) {
            const auto xval = xs(row, split_col);
            const auto yval = ys[row];
            const auto yhat = xval <= split_val ? yhat_l : yhat_r;
            err += pow((yval - yhat), 2.0);
        }

        return (std::isnan(err) ? doubleMax : err);
    }

    template <typename MatrixT, typename VectorT>
    SplitPoint
    best_split(const Dataset<MatrixT, VectorT>& data, const SizeVec& filter) {
        // TODO min size guard
        size_t best_col = 0;
        float best_val = floatNaN;
        double best_err = doubleMax;

        std::vector< std::future<double> > futures;

        const auto ncols = data.ncol();
        for (size_t col = 0; col != ncols; ++col) {
            const auto uniques = data.unique_x(col, filter);
            const auto uniq_sz = uniques.size();
            if (uniq_sz < 2) {
                continue;
            }

            futures.clear();

            std::transform(
                uniques.begin(),
                uniques.end(),
                std::back_inserter(futures),
                [col, &data, &filter](const float value) {
                    return std::async(
                        std::launch::deferred,
                        [col, &data, &filter, value] () {
                            return calc_total_err(col, value, data, filter);
                        });
                });

            for (size_t idx = 0; idx != uniq_sz; ++idx) {
                // total squared error for left and right side of split_val
                const auto err = futures[idx].get();

                // TODO randomly allow the same error as best to 'win'
                if (err < best_err) {
                    best_col = col;
                    best_val = uniques[idx];
                    best_err = err;
                }
            }
        }
        return SplitPoint(best_col, best_val);
    }
}
#endif //KMBNW_ODVB_SPLITPOINT_H
