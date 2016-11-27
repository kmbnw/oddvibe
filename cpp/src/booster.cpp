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
#include "booster.h"
#include "regression_tree.h"
#include "ecdf_sampler.h"

namespace oddvibe {
    Booster::Booster(
            const size_t& ncols,
            const size_t& depth,
            const std::function<double(const std::vector<float>&, const std::vector<float>&)> &err_fn) :
            m_ncols(ncols), m_depth(depth), m_seed(time(0)), m_err_fn(err_fn) {
    }

    void Booster::update_one(const std::vector<float> &xs, const std::vector<float> &ys) const {
        // set up initial uniform distribution
        const size_t len = ys.size();
        std::vector<float> pmf(len, 100.0 / len);

        EmpiricalSampler sampler(m_seed, pmf);

        Partitioner builder(m_ncols, m_depth, m_err_fn, xs, ys);
        builder.build(sampler);

        const RegressionTree tree(builder);

        std::vector<float> yhats;
        tree.predict(xs, yhats);

        for (size_t k = 0; k < len; ++k) {
        }
    }
}
