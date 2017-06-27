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

#include <cmath>
#include "split_point.h"

namespace oddvibe {
    SplitPoint::SplitPoint(const size_t split_col, const float split_val):
        m_split_col(split_col),
        m_split_val(split_val) {
    }

    bool SplitPoint::is_valid() const {
        return !std::isnan(m_split_val);
    }

    float SplitPoint::split_val() const {
        return m_split_val;
    }

    size_t SplitPoint::split_col() const {
        return m_split_col;
    }
}
