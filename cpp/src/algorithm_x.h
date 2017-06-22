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

#ifndef KMBNW_ODVB_ALGORITHM_X
#define KMBNW_ODVB_ALGORITHM_X

#include <unordered_set>
#include "defs_x.h"

namespace oddvibe {
    SizeVec sequential_ints(const size_t len);

    void update_counts(const SizeVec& src, SizeVec& counts);

    template <typename MatrixT>
    std::unordered_set<float> unique_x(
            const MatrixT& xs,
            const size_t col,
            const SizeVec& indices) {
        std::unordered_set<float> uniques;

        for (const auto & row : indices) {
            uniques.insert(xs(row, col));
        }
        return uniques;
    }
}
#endif //KMBNW_ODVB_ALGORITHM_X
