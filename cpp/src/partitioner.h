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
#include "sampler.h"

#ifndef KMBNW_ODVB_PRT_H
#define KMBNW_ODVB_PRT_H

namespace oddvibe {

    double rmse(const std::vector<float> &left, const std::vector<float> &right);
    float filtered_mean(const std::vector<float> &ys, const std::vector<bool> &row_filter);

    /**
     * Builder class for decision trees.
     */
    class Partitioner {
        public:
            Partitioner(
                const size_t& ncols,
                const size_t& max_depth,
                const std::vector<float> &xs,
                const std::vector<float> &ys,
                const std::function<double(const std::vector<float>&, const std::vector<float>&)> &err_fn = rmse);

            Partitioner(const Partitioner& other) = delete;
            Partitioner& operator=(const Partitioner& other) = delete;

            void build(Sampler& sampler);

            std::vector<size_t> m_feature_idxs;
            std::vector<float> m_split_vals;
            std::unordered_map<size_t, float> m_predictions;

            const size_t m_ncols;


        private:
            const size_t m_tree_sz;
            const std::vector<float> m_xs;
            const std::vector<float> m_ys;
            const std::function<double(const std::vector<float>&, const std::vector<float>&)> m_err_fn;
    
            void set_row_filter(
                const std::vector<bool> &row_filter,
                std::vector<bool> &tmp_filter,
                size_t feature_idx,
                const float &split_value,
                bool left) const;

            void build(Sampler& sampler, const size_t &node_idx, const std::vector<bool> &row_filter);
    };
}
#endif //KMBNW_ODVB_PRT_H
