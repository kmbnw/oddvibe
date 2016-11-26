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

#ifndef KMBNW_ODVB_RTB_H
#define KMBNW_ODVB_RTB_H

namespace oddvibe {

    class Partitioner {
        public:
            Partitioner(
                const size_t &ncols,
                const std::function<double(const std::vector<float>&, const std::vector<float>&)> &err_fn,
                const std::vector<float> &xs,
                const std::vector<float> &ys);

            void find_splits(
                const size_t &depth,
                std::vector<size_t> &feature_idxs,
                std::vector<float> &split_vals) const;

        private:
            size_t _ncols;
            std::function<double(const std::vector<float>&, const std::vector<float>&)> _err_fn;
            std::vector<float> _xs;
            std::vector<float> _ys;
    
            void set_row_filter(
                    std::vector<bool> &row_filter,
                    const int &feature_idx,
                    const float &split_value,
                    bool left) const;

            void find_splits(
                const size_t &depth,
                std::vector<bool> &row_filter,
                std::vector<size_t> &feature_idxs,
                std::vector<float> &split_vals) const;
    };

    double split_error(const std::vector<float> &left, const std::vector<float> &right);
}
#endif //KMBNW_ODVB_RTB_H
