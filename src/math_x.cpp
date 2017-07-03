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

#include <cmath>
#include "math_x.h"

namespace oddvibe {
    void normalize(std::vector<float>& pmf) {
        const auto norm = std::accumulate(std::begin(pmf), std::end(pmf), 0.0);
        std::transform(
            std::begin(pmf),
            std::end(pmf),
            std::begin(pmf),
            [norm](float f) { return f / norm; });
    }

    std::vector<float>
    divide_vector(const std::vector<size_t>& seq, const size_t divisor) {
        if (divisor == 0) {
            throw std::invalid_argument("Divisor cannot be zero");
        }
        const auto f_divisor = (1.0f * divisor) + 1;
        std::vector<float> norm_seq(seq.size(), 0);

        std::transform(
            seq.begin(),
            seq.end(),
            norm_seq.begin(),
            [f_divisor](const size_t count) {
                const auto norm_count = (1.0 * count) / f_divisor;
                if (std::isnan(norm_count)) {
                    throw std::logic_error("NaN for divided values");
                }
                return norm_count;
            });
        return norm_seq;
    }
}
