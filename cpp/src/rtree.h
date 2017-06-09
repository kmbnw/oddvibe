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
#include "split_data.h"
#include "train_data.h"

namespace oddvibe {

    /**
     * Regression tree.
     */
    class RTree {
        public:
            RTree(const DataSet& data);
            RTree(const DataSet& data, const std::vector<bool>& active);

            /**
             * No copy.
             */
            RTree(const RTree& other) = delete;
            /**
             * No copy.
             */
            RTree& operator=(const RTree& other) = delete;

            SplitData
            best_split(const DataSet& data) const;

            float RTree::predict(const DataSet& data) const;

        private:
            std::vector<bool> m_active;
            std::unique_ptr<RTree> m_left_child;
            std::unique_ptr<RTree> m_right_child;

            double
            calc_total_err(
                const DataSet& data,
                const size_t col,
                const float split,
                const float yhat_l,
                const float yhat_r) const;

            std::pair<float, float>
            predict_split(
                const DataSet& data,
                const size_t col,
                const float split) const;

            void populate_filter(
                    const DataSet& data,
                    const size_t col,
                    const float split_val,
                    std::vector<bool> left_filter,
                    std::vector<bool> right_filter);
    };
}
#endif //KMBNW_RTREE_H
