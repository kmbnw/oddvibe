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
#include "seq_sampler.h"
#include "cached_sampler.h"
#include "math_x.h"

namespace oddvibe {
    void add_counts(Sampler& sampler, std::vector<unsigned int>& counts) {
        for (size_t k = 0; k != counts.size(); ++k) {
            const size_t s = sampler.next_sample();
            counts[s]++;
        }
    }

    Booster::Booster(Partitioner& builder, const size_t& seed) :
        m_builder(builder), m_seed(seed) {
    }

    void Booster::update_one(
            std::vector<float>& pmf,
            std::vector<unsigned int>& counts,
            const std::vector<float> &xs,
            const std::vector<float> &ys) const {
        // set up initial uniform distribution over all instances
        const size_t nrows = ys.size();
        if (counts.size() != nrows) {
            counts.clear();
            counts.resize(nrows, 0);
        }

        if (pmf.size() != nrows) {
            pmf.clear();
            pmf.resize(nrows, 1.0 / nrows);

            SequentialSampler sampler(0, nrows);
            add_counts(sampler, counts);
            m_builder.build(sampler);
        } else {
            EmpiricalSampler sampler(m_seed, pmf);
            CachedSampler cache(sampler);
            add_counts(cache, counts);
            m_builder.build(cache);
        }

        const RegressionTree tree(m_builder);

        std::vector<float> yhats;
        tree.predict(xs, yhats);

        std::vector<double> loss(yhats.size(), 0);
        std::transform(yhats.begin(), yhats.end(), ys.begin(), loss.begin(),
            [](float yhat, float y) { return pow(yhat - y, 2); });

        const double max_loss = *std::max_element(loss.begin(), loss.end());

        double epsilon = 0.0;
        for (size_t k = 0; k != loss.size(); ++k) {
            epsilon += pmf[k] * loss[k];
        }

        const double beta = epsilon / (max_loss - epsilon);

        for (size_t k = 0; k != loss.size(); ++k) {
            const double d = loss[k] / max_loss;
            pmf[k] = (float) (pow(beta, 1 - d) * pmf[k]);
        }
        normalize(pmf);
    }
}
