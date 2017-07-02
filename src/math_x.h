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
#include "defs_x.h"

#ifndef KMBNW_ODVB_MATHX_H
#define KMBNW_ODVB_MATHX_H

namespace oddvibe {
    /**
     * Normalize a vector to sum to 1 (e.g. proper probability mass function).
     * @param[inout] pmf The vector to normalize; overwritten in-place.
     */
    void normalize(FloatVec& pmf);

    template <typename VectorT, typename IteratorT>
    double
    mean(const VectorT& seq, const IteratorT first, const IteratorT last) {
        if (first == last) {
            return 0;
        }

        size_t count = 0;
        double total = 0;

        for (auto row = first; row != last; row = std::next(row)) {
            total = total + (seq[*row] - total) / (++count);
        }
        return total;
    }

    template <typename VectorT, typename IteratorT>
    double
    variance(const VectorT& seq, const IteratorT first, const IteratorT last) {
        if (first == last) {
            return doubleNaN;
        }

        size_t count = 0;
        double total = 0;
        const auto avg_x = mean(seq, first, last);

        for (auto row = first; row != last; row = std::next(row)) {
            total += pow(seq[*row] - avg_x, 2);
            ++count;
        }

        return (count < 1 ? doubleNaN : total / count);
    }

    template <typename VectorTLeft, typename VectorTRight>
    DoubleVec loss_seq(const VectorTLeft& ys, const VectorTRight& yhats) {
        if (ys.size() != yhats.size()) {
            throw std::logic_error("Observed and predicted must be same size");
        }
        DoubleVec loss(yhats.size(), 0);
        std::transform(
            yhats.begin(),
            yhats.end(),
            ys.begin(),
            loss.begin(),
            [](const double predicted, const double observed) {
                return pow(predicted - observed, 2);
            });
        return loss;
    }

    FloatVec normalize_counts(const SizeVec &counts, const size_t nrounds);
}
#endif //KMBNW_ODVB_MATHX_H
