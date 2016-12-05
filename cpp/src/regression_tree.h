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

#ifndef KMBNW_ODVB_DECISIONTREE_H
#define KMBNW_ODVB_DECISIONTREE_H

namespace oddvibe {
    class RegressionTree {
        public:
            RegressionTree(const Partitioner& builder);

            RegressionTree(const RegressionTree& other) = delete;
            RegressionTree& operator=(const RegressionTree& other) = delete;

            void predict(const std::vector<float> &xs, std::vector<float> &yhats) const;

        private:
            const size_t m_ncols;
            const std::vector<size_t>& m_feature_idxs;
            const std::vector<float>& m_split_vals;
            const std::unordered_map<size_t, float>& m_predictions;
    };
}
#endif //KMBNW_ODVB_DECISIONTREE_H
