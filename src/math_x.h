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
#include <algorithm>
#include <stdexcept>

#ifndef KMBNW_ODVB_MATHX_H
#define KMBNW_ODVB_MATHX_H

/*! \file */

/**
 * Namespace functions
 */
namespace oddvibe {
    /**
     * Normalize a vector to sum to 1 (e.g. proper probability mass function).
     *
     * \param pmf[inout] The vector to normalize; overwritten in-place.
     */
    void normalize(std::vector<float>& pmf);

    /**
     * Divide a vector by a scalar.
     *
     * That is, for each element of `seq`, divide it by `divisor'.
     *
     * \param seq The vector of values to divide.
     * \param divisor The value to divide by.  If zero, this will throw an
     * exception.
     * \return A new vector of values divided as described above.
     */
    std::vector<float>
    divide_vector(const std::vector<size_t>& seq, const size_t divisor);

    template <typename FloatT>
    FloatT rolling_mean(FloatT current, FloatT nextval, size_t& count) {
        return current + (nextval - current) / (++count);
    }

    /**
     * Calculate the filtered mean of a vector of values.
     *
     * The elements from the range `[first, last]` are used to filter
     * the input data; they are row indices into the seq values that will be
     * used to calculate the mean.  The row indexes must be within
     * `[0, seq.size())`; this will throw an exception if they are not.
     *
     * \param seq The vector to calculate the filtered mean for.
     * \param first InputIterator to the initial position of
     * the row indexes.
     * \param last InputIterator to the final position of
     * the row indexes.
     * \return The mean of the values in the `seq` vector.  If there are no
     * values to compute (because first == last) this will return 0.
     */
    template <typename FloatT, typename InputIterator>
    FloatT mean(
            const std::vector<FloatT>& seq,
            const InputIterator first,
            const InputIterator last) {
        if (first == last) {
            return 0;
        }

        size_t count = 0;
        FloatT total = 0;
        const auto sz = seq.size();

        for (auto row = first; row != last; row = std::next(row)) {
            const auto idx = *row;
            if (idx >= sz) {
                throw std::out_of_range("Row not in range");
            }
            total = rolling_mean(total, seq[idx], count);
        }
        return total;
    }

    /**
     * Mean-squared error.
     */
    template <typename FloatT>
    double mse_err(const FloatT predicted, const FloatT observed) {
        return pow(predicted - observed, 2.0);
    }

   /**
     * Calculate the filtered variance of a vector of values.
     *
     * The elements from the range `[first, last]` are used to filter
     * the input data; they are row indices into the seq values that will be
     * used to calculate the mean.  The row indexes must be within
     * `[0, seq.size())`; this will throw an exception if they are not.
     *
     * \param seq The vector to calculate the filtered variance for.
     * \param first InputIterator to the initial position of
     * the row indexes.
     * \param last InputIterator to the final position of
     * the row indexes.
     * \return The variance of the values in the `seq` vector.  If there are no
     * values to compute (because e.g. first == last) this will return an
     * appropriate NaN value that can be checked with `std::isnan`.
     */
    template <typename FloatT, typename IteratorT>
    FloatT variance(
            const std::vector<FloatT>& seq,
            const IteratorT first,
            const IteratorT last) {
        constexpr auto nan_val = std::numeric_limits<FloatT>::quiet_NaN();
        if (first == last) {
            return nan_val;
        }

        size_t count = 0;
        FloatT total = 0;
        // mean() will do our range checking, so don't repeat below
        const auto avg_x = mean<FloatT>(seq, first, last);

        for (auto row = first; row != last; row = std::next(row)) {
            total += mse_err(seq[*row], avg_x);
            ++count;
        }

        return (count < 1 ? nan_val : total / count);
    }

    template <typename FloatT>
    std::vector<double>
    loss_seq(const std::vector<FloatT>& ys, const std::vector<FloatT>& yhats) {
        if (ys.size() != yhats.size()) {
            throw std::logic_error("Observed and predicted must be same size");
        }
        std::vector<double> loss(yhats.size(), 0);
        std::transform(
            yhats.begin(),
            yhats.end(),
            ys.begin(),
            loss.begin(),
            mse_err<FloatT>);
        return loss;
    }
}
#endif //KMBNW_ODVB_MATHX_H
