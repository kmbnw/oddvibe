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
#include <unordered_map>
#include <functional>
#include "partitioner.h"

#ifndef KMBNW_ODVB_MATHX_H
#define KMBNW_ODVB_MATHX_H

namespace oddvibe {
    /**
     * Normalize a vector to sum to 1 (e.g. proper probability mass function).
     * @param[inout] pmf The vector to normalize; overwritten in-place.
     */
    void normalize(std::vector<float>& pmf);
}
#endif //KMBNW_ODVB_MATHX_H
