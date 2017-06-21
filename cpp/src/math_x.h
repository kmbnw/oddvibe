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
#include <unordered_set>
#include "defs_x.h"

#ifndef KMBNW_ODVB_MATHX_H
#define KMBNW_ODVB_MATHX_H

namespace oddvibe {
    /**
     * Normalize a vector to sum to 1 (e.g. proper probability mass function).
     * @param[inout] pmf The vector to normalize; overwritten in-place.
     */
    void normalize(FloatVec& pmf);

    double rmse_loss(const float predicted, const float observed);

    double mean(
        const FloatVec seq,
        const SizeConstIter first,
        const SizeConstIter last);

    double variance(
        const FloatVec seq,
        const SizeConstIter first,
        const SizeConstIter last);

    DoubleVec loss_seq(const FloatVec& ys, const FloatVec& yhats);

    template<typename VectorType, typename IteratorType>
    std::pair<float, float>
    partitioned_mean(
            const VectorType& ys,
            const IteratorType first,
            const IteratorType pivot,
            const IteratorType last) {

        // std::optional would be good here, but not at C++ 17 yet
        // for this install
        if (first == last || pivot == first || pivot == last) {
            return std::make_pair(floatNaN, floatNaN);
        }

        const float yhat_l = mean(ys, first, pivot);
        if (std::isnan(yhat_l)) {
            return std::make_pair(floatNaN, floatNaN);
        }

        const float yhat_r = mean(ys, pivot, last);
        if (std::isnan(yhat_r)) {
            return std::make_pair(floatNaN, floatNaN);
        }
        return std::make_pair(yhat_l, yhat_r);
    }
}
#endif //KMBNW_ODVB_MATHX_H
