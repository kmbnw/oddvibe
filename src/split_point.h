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
#include "math_x.h"
#include "dataset.h"

namespace oddvibe {
    /**
     * Regression tree split point.
     */
    template <typename FloatT>
    class SplitPoint {
        public:
            SplitPoint<FloatT>() = default;

            SplitPoint<FloatT>(const size_t split_col, const FloatT split_val) :
                m_split_col(split_col),
                m_split_val(split_val) { }

            SplitPoint<FloatT>(SplitPoint<FloatT>&& other) = default;
            SplitPoint<FloatT>(const SplitPoint<FloatT>& other) = default;
            SplitPoint<FloatT>& operator=(
                const SplitPoint<FloatT>& other) = default;
            SplitPoint<FloatT>& operator=(SplitPoint<FloatT>&& other) = default;
            ~SplitPoint<FloatT>() = default;

            /**
             * \return The value at which this split point occurred.
             */
            FloatT split_val() const {
                return m_split_val;
            }

            /**
             * \return The zero based feature column at which this split point
             * occurred.
             */
            size_t split_col() const {
                return m_split_col;
            }

            /**
             * \return True if this instance has a non-NaN split_val().
             */
            bool is_valid() const {
                return !std::isnan(m_split_val);
            }

            /**
             * Partition the input sequence according to this instance.
             *
             * Rearranges the elements from the range `[first, last]` in such a
             * way that all the elements for which
             *     `mat(row, split_col()) <= split_val()`
             * returns true precede all those for which it returns false.  The
             * iterator returned points to the first element of the second group.
             * \param mat Numeric matrix of feature data
             * \param first BidirectionalIterator to the initial position of
             * the row indexes.
             * \param last BidirectionalIterator to the final position of
             * the row indexes.
             * \return Iterator as described in the main description.
             * \sa split_col()
             * \sa split_val()
             */
            template <typename BidirectionalIterator>
            BidirectionalIterator
            partition_idx(
                const FloatMatrix<FloatT>& mat,
                BidirectionalIterator first,
                BidirectionalIterator last)
            const {
                return std::partition(
                    first,
                    last,
                    [this, &mat](const size_t row){
                        return mat(row, this->m_split_col) <= this->m_split_val;
                    });
            }

        private:
            size_t m_split_col = 0;
            FloatT m_split_val = std::numeric_limits<FloatT>::quiet_NaN();
    };

    /**
     * Create a new "best" SplitPoint.
     *
     * "Best" means that for a given feature matrix and rows to consider in
     * that matrix, a binary split done on the feature column and feature value
     * produce the lowest total error.  The lowest total error may not (and
     * probably is not) unique; this function chooses the first such column and
     * value that it finds that fulfills the criteria.
     *
     * \param data Input feature matrix.
     * \param filter The rows of the feature matrix to consider when finding
     * the best split column and value.
     * \return A new SplitPoint instance that contains the best-split selection.
     * If no such split could be found (due to lack of unique values, etc)
     * then the value of is_valid() from the returned SplitPoint will be false.
     */
    template <typename FloatT>
    SplitPoint<FloatT>
    best_split(const Dataset<FloatT>& data, const std::vector<size_t>& filter) {
        // TODO min size guard
        size_t best_col = 0;
        FloatT best_val = std::numeric_limits<FloatT>::quiet_NaN();
        double best_err = std::numeric_limits<double>::max();

        std::vector< std::future<double> > futures;

        const auto ncols = data.ncol();

        const auto err_fn = [&data, &filter] (
                const size_t col, const FloatT value) {
            return data.calc_total_err(col, value, filter.begin(), filter.end());
        };

        for (size_t col = 0; col != ncols; ++col) {
            const auto uniques = data.unique_x(col, filter.begin(), filter.end());
            const auto uniq_sz = uniques.size();
            if (uniq_sz < 2) {
                continue;
            }

            futures.clear();

            std::transform(
                uniques.begin(),
                uniques.end(),
                std::back_inserter(futures),
                [&err_fn, col](const FloatT value) {
                    return std::async(std::launch::deferred, err_fn, col, value);
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
        return SplitPoint<FloatT>(best_col, best_val);
    }
}
#endif //KMBNW_ODVB_SPLITPOINT_H
