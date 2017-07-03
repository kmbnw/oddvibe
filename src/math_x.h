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

namespace oddvibe {
    /**
     * Normalize a vector to sum to 1 (e.g. proper probability mass function).
     * @param[inout] pmf The vector to normalize; overwritten in-place.
     */
    void normalize(std::vector<float>& pmf);

    template <typename FloatT>
    FloatT rolling_mean(FloatT current, FloatT nextval, size_t& count) {
        return current + (nextval - current) / (++count);
    }

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

        for (auto row = first; row != last; row = std::next(row)) {
            total = rolling_mean(total, seq[*row], count);
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

    template <typename FloatT, typename IteratorT>
    FloatT variance(
            const std::vector<FloatT>& seq,
            const IteratorT first,
            const IteratorT last) {
        const auto nan_val = std::numeric_limits<FloatT>::quiet_NaN();
        if (first == last) {
            return nan_val;
        }

        size_t count = 0;
        FloatT total = 0;
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

    std::vector<float>
    normalize_counts(const std::vector<size_t>& counts, const size_t nrounds);
}
#endif //KMBNW_ODVB_MATHX_H
