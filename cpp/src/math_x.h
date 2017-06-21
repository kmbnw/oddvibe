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
}
#endif //KMBNW_ODVB_MATHX_H
