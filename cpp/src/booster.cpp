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
#include "cached_sampler.h"

namespace oddvibe {
    void normalize(std::vector<float>& pmf) {
        double norm = std::accumulate(pmf.begin(), pmf.end(), 0.0f);
        for (size_t k = 0; k != pmf.size(); ++k) {
            pmf[k] = (float) (pmf[k] / norm);
        }
    }

    void add_counts(Sampler& sampler, std::vector<unsigned int>& counts) {
        for (size_t k = 0; k != counts.size(); ++k) {
            counts[sampler.next_sample()]++;
        }
    }

    Booster::Booster(
            Partitioner* const builder,
            const std::function<double(const std::vector<float>&, const std::vector<float>&)> &err_fn) :
            m_builder(builder), m_seed(time(0)), m_err_fn(err_fn) {
    }

    void Booster::fit(
            const size_t& num_rounds,
            const std::vector<float> &xs,
            const std::vector<float> &ys) const {
        // set up initial uniform distribution over all instances
        const size_t nrows = ys.size();
        std::vector<float> pmf(nrows, 100.0 / nrows);
        std::vector<unsigned int> counts(nrows, 0);
        std::vector<float> yhats;
        std::vector<double> loss;

        for (size_t round = 0; round < num_rounds; ++round) {
            loss.clear();
            yhats.clear();

            EmpiricalSampler sampler(m_seed, pmf);
            CachedSampler cache(sampler);
            add_counts(cache, counts);

            m_builder->build(cache);
            const RegressionTree tree((*m_builder));

            tree.predict(xs, yhats);
            loss.resize(yhats.size());

            std::transform(yhats.begin(), yhats.end(), ys.begin(), loss.begin(),
                [](float yhat, float y) { return pow(yhat - y, 2); });

            const double max_loss = *std::max_element(loss.begin(), loss.end());

            // TODO use the error function
            double epsilon = 0.0;
            for (size_t k = 0; k != loss.size(); ++k) {
                epsilon += pmf[k] * loss[k];
            }

            double error_diff = 1e-6;
            for (size_t k = 0; k != loss.size(); ++k) {
                error_diff = std::max(error_diff, loss[k] - epsilon);
            }

            const double beta = epsilon / error_diff;

            for (size_t k = 0; k != loss.size(); ++k) {
                const double d = loss[k] / max_loss;
                pmf[k] = (float) (pow(beta, 1 - d) * pmf[k]);
            }
            normalize(pmf);
        }
    }
}
