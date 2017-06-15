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

#ifndef KMBNW_RTREE_H
#define KMBNW_RTREE_H

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include "split_data.h"
#include "train_data.h"

namespace oddvibe {

    /**
     * Regression decision tree
     */
    class RTree {
        public:
            RTree() = default;
            RTree(RTree&& other) = default;
            RTree(const RTree& other) = default;
            RTree& operator=(const RTree& other) = default;
            ~RTree() = default;

            FloatVec predict(const DataSet& data) const;

            void fit(const DataSet& data, const SizeVec& filter);

            SplitData
            best_split(const DataSet& data, const SizeVec& filter) const;

        private:
            float m_yhat = std::numeric_limits<float>::quiet_NaN();
            bool m_is_leaf = true;
            SplitData m_split;

            std::unique_ptr<RTree> m_left;
            std::unique_ptr<RTree> m_right;

            void predict(
                const DataSet& data,
                const BoolVec& filter,
                FloatVec& yhat)
            const;

            void fill_filter(
                const DataSet& data,
                const BoolVec& init_filter,
                BoolVec& left_filter,
                BoolVec& right_filter)
            const;

            double calc_total_err(
                const DataSet& data,
                const SplitData& split,
                const SizeVec& filter,
                const std::pair<float, float>& yhat)
            const;

            std::pair<float, float> fit_children(
                const DataSet& data,
                const SplitData&,
                const SizeVec& filter)
            const;

            void fill_row_idx(
                const DataSet& data,
                const SplitData& split,
                const SizeVec& filter,
                SizeVec& left_rows,
                SizeVec& right_rows)
            const;
    };
}
#endif //KMBNW_RTREE_H
