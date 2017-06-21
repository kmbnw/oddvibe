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
#include <cmath>
#include "math_x.h"

namespace oddvibe {
    void normalize(FloatVec& pmf) {
        const auto norm = std::accumulate(std::begin(pmf), std::end(pmf), 0.0);
        std::transform(
            std::begin(pmf),
            std::end(pmf),
            std::begin(pmf),
            [norm = norm](float f) { return f / norm; });
    }

    double rmse_loss(const float predicted, const float observed) {
        return pow(predicted - observed, 2);
    }


    double mean(
            const FloatVec seq,
            const SizeConstIter first,
            const SizeConstIter last) {
        if (first == last) {
            return 0;
        }

        size_t count = 0;
        double total = 0;

        for (auto row = first; row != last; row = std::next(row)) {
            total += seq[*row];
            ++count;
        }
        return (count < 1 ? 0 : total / count);
    }

    double variance(
            const FloatVec seq,
            const SizeConstIter first,
            const SizeConstIter last) {
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

    DoubleVec loss_seq(const FloatVec& ys, const FloatVec& yhats) {
        if (ys.size() != yhats.size()) {
            throw std::logic_error("Observed and predicted must be same size");
        }
        DoubleVec loss(yhats.size(), 0);
        std::transform(
            yhats.begin(),
            yhats.end(),
            ys.begin(),
            loss.begin(),
            rmse_loss);
        return loss;
    }

    FloatVec normalize_counts(const SizeVec& counts, const size_t nrounds) {
        const auto f_nrounds = (1.0f * nrounds) + 1;
        FloatVec norm_counts(counts.size(), 0);

        std::transform(
            counts.begin(),
            counts.end(),
            norm_counts.begin(),
            [f_nrounds = f_nrounds](const size_t count) {
                const auto norm_count = (1.0 * count) / f_nrounds;
                if (std::isnan(norm_count)) {
                    throw std::logic_error("NaN for normalized count");
                }
                return norm_count;
            });
        return norm_counts;
    }
}
