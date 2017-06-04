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
#include "split_data.h"

namespace oddvibe {
    SplitData::SplitData(
            const float value,
            const size_t col_idx,
            const double total_err):
        m_value(value), m_col_idx(col_idx), m_total_err(total_err) {}

    bool SplitData::is_valid() const {
        return !std::isnan(m_value);
    }
   
    float SplitData::value() const {
        return m_value;
    }

    size_t SplitData::col_idx() const {
        return m_col_idx;
    }

    double SplitData::total_err() const {
        return m_total_err;
    }
}
