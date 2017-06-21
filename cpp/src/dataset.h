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
#include "defs_x.h"
#include "math_x.h"
#include "split_point.h"

#ifndef KMBNW_DATASET_H
#define KMBNW_DATASET_H

namespace oddvibe {
    template<typename MatrixType, typename VectorType>
    class Dataset {
        public:
            Dataset(const MatrixType& xs, const VectorType& ys) {
                if (xs.nrows() != ys.size()) {
                    throw std::invalid_argument("Mismatched number of rows");
                }
                m_xs = xs;
                m_ys = ys;
            }

            Dataset(MatrixType&& xs, VectorType&& ys) {
                if (xs.nrows() != ys.size()) {
                    throw std::invalid_argument("Mismatched number of rows");
                }
                m_xs = std::move(xs);
                m_ys = std::move(ys);
            }

            // TODO explicit defaults

            // total squared error for left and right side of split_val
            double calc_total_err(
                    const SplitPoint& split,
                    const SizeVec& filter) const {

                if (filter.empty()) {
                    return doubleNaN;
                }

                const auto avg = partitioned_mean(split, filter);
                const float yhat_l = avg.first;
                const float yhat_r = avg.second;

                if (std::isnan(yhat_l) || std::isnan(yhat_r)) {
                    return doubleNaN;
                }

                const auto split_val = split.split_val();
                const auto split_col = split.split_col();
                double err = 0;
                for (const auto & row : filter) {
                    const auto x_j = m_xs(row, split_col);
                    const auto y_j = m_ys[row];
                    const auto yhat_j = x_j <= split_val ? yhat_l : yhat_r;
                    err += pow((y_j - yhat_j), 2.0);
                }
                return err;
            }

            size_t nrows() const {
                return m_xs.nrows();
            }

            size_t ncols() const {
                return m_xs.ncols();
            }

            const MatrixType& xs() const {
                return m_xs;
            }

            const VectorType& ys() const {
                return m_ys;
            }

        private:
            MatrixType m_xs;
            VectorType m_ys;

            std::pair<float, float>
            partitioned_mean(
                    const SplitPoint& split,
                    const SizeVec& filter) const {

                SizeVec part(filter);
                const auto pivot = split.partition_idx(m_xs, part);

                auto first_p = part.begin();
                auto last_p = part.end();
                // std::optional would be good here, but not at C++ 17 yet
                // for this install
                if (first_p == last_p || pivot == first_p || pivot == last_p) {
                    return std::make_pair(floatNaN, floatNaN);
                }

                const float yhat_l = mean(m_ys, first_p, pivot);
                if (std::isnan(yhat_l)) {
                    return std::make_pair(floatNaN, floatNaN);
                }

                const float yhat_r = mean(m_ys, pivot, last_p);
                if (std::isnan(yhat_r)) {
                    return std::make_pair(floatNaN, floatNaN);
                }
                return std::make_pair(yhat_l, yhat_r);
            }
    };
}
#endif //KMBNW_DATASET_H
