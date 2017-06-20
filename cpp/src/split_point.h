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
#ifndef KMBNW_SPLIT_POINT_H
#define KMBNW_SPLIT_POINT_H

#include <cstddef>
#include <limits>
#include "defs_x.h"
#include "dataset.h"

namespace oddvibe {
    class SplitPoint {
        public:
            SplitPoint() = default;

            SplitPoint(const float split_val, const size_t split_col);

            SplitPoint(SplitPoint&& other) = default;
            SplitPoint(const SplitPoint& other) = default;
            SplitPoint& operator=(const SplitPoint& other) = default;
            SplitPoint& operator=(SplitPoint&& other) = default;
            ~SplitPoint() = default;

            float split_val() const;

            size_t split_col() const;

            bool is_valid() const;

            std::pair<BoolVec, BoolVec>
            partition_rows(const FloatMatrix& mat, const BoolVec& filter)
            const;

            std::pair<SizeVec, SizeVec>
            partition_rows(const FloatMatrix& mat, const SizeVec& filter)
            const;

            SizeIter
            partition_idx(
                const FloatMatrix& mat, SizeVec& filter) const;

            double calc_total_err(
                const FloatMatrix& mat,
                const FloatVec& ys,
                const SizeVec& filter) const;

        private:
            float m_split_val = std::numeric_limits<float>::quiet_NaN();
            size_t m_split_col = 0;
    };
}
#endif //KMBNW_SPLIT_POINT_H
