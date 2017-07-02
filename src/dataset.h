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
#ifndef KMBNW_ODVB_DATASET_H
#define KMBNW_ODVB_DATASET_H

#include <cstddef>
#include <utility>

namespace oddvibe {
    template <typename MatrixT, typename VectorT>
    class Dataset {
        public:
            explicit Dataset<MatrixT, VectorT>(MatrixT& xs, VectorT& ys) {
                if (xs.nrow() != ys.size()) {
                    throw std::logic_error("X and Y row counts do not match");
                }
                m_xs = xs;
                m_ys = ys;
            }

            explicit Dataset<MatrixT, VectorT>(MatrixT&& xs, VectorT&& ys) {
                if (xs.nrow() != ys.size()) {
                    throw std::logic_error("X and Y row counts do not match");
                }
                m_xs = std::move(xs);
                m_ys = std::move(ys);
            }

            Dataset<MatrixT, VectorT>(Dataset<MatrixT, VectorT>&& other) = default;
            Dataset<MatrixT, VectorT>(const Dataset<MatrixT, VectorT>& other) = default;
            Dataset<MatrixT, VectorT>& operator=(const Dataset<MatrixT, VectorT>& other) = default;
            Dataset<MatrixT, VectorT>& operator=(Dataset<MatrixT, VectorT>&& other) = default;
            ~Dataset<MatrixT, VectorT>() = default;

            const MatrixT& xs() const { return m_xs; }
            const VectorT& ys() const { return m_ys; }

            FloatVec
            unique_x(const size_t col, const SizeVec& indices) const {
                std::unordered_set<float> uniques;

                for (const auto & row : indices) {
                    uniques.insert(m_xs(row, col));
                }
                return FloatVec(uniques.begin(), uniques.end());
            }

             // total squared error for left and right side of split_val
            double calc_total_err(
                    const size_t split_col,
                    const float split_val,
                    const SizeVec& filter) const {
                size_t count_l = 0;
                size_t count_r = 0;
                double yhat_l = 0;
                double yhat_r = 0;

                for (const auto & row : filter) {
                    if (m_xs(row, split_col) <= split_val) {
                        yhat_l += m_ys[row];
                        ++count_l;
                    } else {
                        yhat_r += m_ys[row];
                        ++count_r;
                    }
                }

                if (count_l == 0 || count_r == 0) {
                    return doubleMax;
                }
                yhat_l /= count_l;
                yhat_r /= count_r;

                const double err =std::accumulate(
                    filter.begin(),
                    filter.end(),
                    0,
                    [split_col, split_val, yhat_l, yhat_r, this](
                            const double init, const size_t row) {
                        const double yhat = (m_xs(row, split_col) <= split_val)
                        ? yhat_l
                        : yhat_r;
                        return init + pow((m_ys[row] - yhat), 2.0);
                    }
                );

                return (std::isnan(err) ? doubleMax : err);
            }

            size_t ncol() const {
                return m_xs.ncol();
            }

            size_t nrow() const {
                return m_xs.nrow();
            }

        private:
            MatrixT m_xs;
            VectorT m_ys;
    };
}
#endif //KMBNW_ODVB_DATASET_H
