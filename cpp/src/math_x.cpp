/*
 * Copyright 2016 Krysta M Bouzek
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
#include <cmath>
#include "math_x.h"

namespace oddvibe {
    void normalize(std::vector<float>& pmf) {
        const auto norm = std::accumulate(std::begin(pmf), std::end(pmf), 0.0);
        std::transform(
            std::begin(pmf),
            std::end(pmf),
            std::begin(pmf),
            [norm = norm](float f) { return f / norm; });
    }
}
