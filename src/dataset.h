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
#include "float_matrix.h"

namespace oddvibe {
    /**
     * Collect feature matrix data and its corresponding response vector.
     *
     * When training we require both input features and the response values;
     * this class exists to simplify use of them inside of models.
     */
    template <typename FloatT>
    class Dataset {
        public:
            /**
             * Construct a new Dataset.
             *
             * Enforces the precondition that `xs.nrow() == ys.size()`.
             *
             * \param xs Feature matrix
             * \param ys Response vector.
             */
            explicit Dataset<FloatT>(
                    const FloatMatrix<FloatT>& xs,
                    const std::vector<FloatT>& ys) {
                if (xs.nrow() != ys.size()) {
                    throw std::logic_error("X and Y row counts do not match");
                }
                m_xs = xs;
                m_ys = ys;
            }

            /**
             * Move-construct a new Dataset.
             *
             * Enforces the precondition that `xs.nrow() == ys.size()`.
             *
             * \param xs Feature matrix
             * \param ys Response vector.
             */
            explicit Dataset<FloatT>(
                    FloatMatrix<FloatT>&& xs,
                    std::vector<FloatT>&& ys) {
                if (xs.nrow() != ys.size()) {
                    throw std::logic_error("X and Y row counts do not match");
                }
                m_xs = std::move(xs);
                m_ys = std::move(ys);
            }

            Dataset<FloatT>(Dataset<FloatT>&& other) = default;
            Dataset<FloatT>(const Dataset<FloatT>& other) = default;
            Dataset<FloatT>& operator=(const Dataset<FloatT>& other) = default;
            Dataset<FloatT>& operator=(Dataset<FloatT>&& other) = default;
            ~Dataset<FloatT>() = default;

            /**
             * Find all unique values for a given feature column.
             *
             * The row indexes considered for finding unique values are taken
             * from the input range `[first, last]`.
             *
             * \param col The zero-based feature column to get unique values for.
             * \param first InputIterator to the initial position of
             * the row indexes.
             * \param last InputIterator to the final position of
             * the row indexes.
             * \return Vector of unique values for the given feature column
             * (in no particular order).
             */
            template <typename InputIterator>
            std::vector<FloatT>
            unique_x(
                const size_t col,
                const InputIterator first,
                const InputIterator last) const {
                std::unordered_set<FloatT> uniques;

                for (auto row = first; row != last; row = std::next(row)) {
                    uniques.insert(m_xs(*row, col));
                }
                return std::vector<FloatT>(uniques.begin(), uniques.end());
            }

            /**
             * Calculate total squared error for a split point.
             *
             * The error is calculated such that the elements from the range
             * `[first, last]` that satisfy
             * `feature_matrix(row, split_col) <= split_val`
             * are used as row indexes for the "left hand" error, and the others
             * are used as row indexes for the "right hand" error.
             *
             * \param split_col The zero-based feature column to split on.
             * \param split_val The value of the feature to split on.
             * \param first ForwardIterator to the initial position of
             * the row indexes.
             * \param last ForwardIterator to the final position of
             * the row indexes.
             * \return Total squared error when splitting on the input split
             * point.
             */
            template <typename ForwardIterator>
            double calc_total_err(
                    const size_t split_col,
                    const FloatT split_val,
                    const ForwardIterator first,
                    const ForwardIterator last) const {
                FloatT yhat_l  = 0, yhat_r  = 0;
                size_t count_l = 0, count_r = 0;

                const auto is_left = [split_col, split_val, this](
                        const size_t row) {
                    return m_xs(row, split_col) <= split_val;
                };

                for (auto row = first; row != last; row = std::next(row)) {
                    // rolling mean
                    if (is_left(*row)) {
                        yhat_l = rolling_mean(yhat_l, m_ys[*row], count_l);
                    } else {
                        yhat_r = rolling_mean(yhat_r, m_ys[*row], count_r);
                    }
                }

                if (count_l == 0 || count_r == 0) {
                    return doubleMax;
                }

                const auto acc_err = [&is_left, yhat_l, yhat_r, this](
                        const double init, const size_t row) {
                    const double yhat = is_left(row) ? yhat_l : yhat_r;
                    return init + pow((m_ys[row] - yhat), 2.0);
                };

                const double err = std::accumulate(first, last, 0, acc_err);

                return (std::isnan(err) ? doubleMax : err);
            }

            /**
             * \return Number of columns in the feature matrix
             */
            size_t ncol() const {
                return m_xs.ncol();
            }

            /**
             * \return Number of rows in the feature matrix
             */
            size_t nrow() const {
                return m_xs.nrow();
            }

            /**
             * \return Feature matrix.
             */
            const FloatMatrix<FloatT>& xs() const {
                return m_xs;
            }

            /**
             * \return Response vector.
             */
            const std::vector<FloatT>& ys() const {
                return m_ys;
            }

        private:
            FloatMatrix<FloatT> m_xs;
            std::vector<FloatT> m_ys;
    };
}
#endif //KMBNW_ODVB_DATASET_H
