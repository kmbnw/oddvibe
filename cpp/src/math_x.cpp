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

#include <vector>
#include <algorithm>
#include <functional>
#include "math_x.h"

namespace oddvibe {
    void normalize(std::vector<float>& pmf) {
        double norm = std::accumulate(pmf.begin(), pmf.end(), 0.0f);
        for (size_t k = 0; k != pmf.size(); ++k) {
            pmf[k] = (float) (pmf[k] / norm);
        }
    }
}
