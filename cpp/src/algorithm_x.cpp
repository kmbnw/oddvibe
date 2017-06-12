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

#include <algorithm>
#include "algorithm_x.h"

namespace oddvibe {
    std::vector<size_t>
    sequential_ints(const size_t len) {
        std::vector<size_t> seq(len, 0);
        std::iota(seq.begin(), seq.end(), 0);
        return seq;
    }
}