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
#ifndef KMBNW_SPLIT_DATA_H
#define KMBNW_SPLIT_DATA_H

#include <cstddef>
#include <limits>
#include "defs_x.h"

namespace oddvibe {
    class SplitData {
        public:
            SplitData() = default;

            SplitData(const float split_val, const size_t split_col);

            SplitData(SplitData&& other) = default;
            SplitData(const SplitData& other) = default;
            SplitData& operator=(const SplitData& other) = default;
            ~SplitData() = default;

            float split_val() const;

            size_t split_col() const;

            bool is_valid() const;

        private:
            float m_split_val = std::numeric_limits<float>::quiet_NaN();
            size_t m_split_col = 0;
    };
}
#endif //KMBNW_SPLIT_DATA_H
