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

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <functional>
#include "regression_tree.h"
#include "partitioner.h"

namespace oddvibe {
    RegressionTree::RegressionTree(const Partitioner &builder) :
        m_ncols(builder.ncols()),
        m_feature_idxs(builder.m_feature_idxs),
        m_split_vals(builder.m_split_vals),
        m_predictions(builder.m_predictions) {
    }

    void RegressionTree::predict(const std::vector<float> &xs, std::vector<float> &yhats) const {
        yhats.clear();

        const size_t tree_sz = m_feature_idxs.size();

        for (size_t row_idx = 0; row_idx != xs.size(); row_idx += m_ncols) {
            // parent at element K, left child at element 2K, right child at element 2K + 1
            size_t k = 1;
            size_t next_k = 1;

            while (true) {
                // check the case of max_depth not being reached
                if (m_predictions.find(k) != m_predictions.end()) {
                    break;
                }
                const float split_val = m_split_vals[k];
                const size_t feature_idx = m_feature_idxs[k];
                const bool go_left = xs[row_idx + feature_idx] <= split_val;
                next_k = go_left ? 2 * k : 2 * k + 1;

                if (next_k < tree_sz) {
                    k = next_k;
                } else {
                    break;
                }
            }

            // at a leaf node so make the prediction
            yhats.push_back(m_predictions.find(next_k)->second);
        }
    }
}
