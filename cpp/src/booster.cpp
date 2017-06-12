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

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <functional>
#include "booster.h"
#include "rtree.h"
#include "ecdf_sampler.h"
#include "math_x.h"

namespace oddvibe {
    void
    update_counts(const std::vector<size_t>& src, std::vector<size_t>& counts) {
        for (const auto & idx : src) {
            ++counts[idx];
        }
    }

    Booster::Booster(const size_t& seed) : m_seed(seed) {
    }

    void
    Booster::update_one(
        const DataSet& data,
        std::vector<float>& pmf,
        std::vector<size_t>& counts)
    const {
        // set up initial uniform distribution over all instances
        const size_t nrows = data.nrows();
        if (counts.size() != nrows) {
            counts.clear();
            counts.resize(nrows, 0);
        }

        if (pmf.size() != nrows) {
            pmf.clear();
            pmf.resize(nrows, 1.0 / nrows);
        }

        EmpiricalSampler sampler(m_seed, pmf);

        const auto active = sampler.gen_samples(nrows);
        update_counts(active, counts);

        RTree::Fitter fitter(active);
        fitter.fit(data);
        const auto tree = fitter.build();
        const auto yhats = tree.predict(data);
        const auto loss = data.loss(yhats);
        const double max_loss = *std::max_element(loss.begin(), loss.end());

        double epsilon = 0.0;
        for (size_t k = 0; k != loss.size(); ++k) {
            epsilon += pmf[k] * loss[k];
        }

        const double beta = epsilon / (max_loss - epsilon);

        if (epsilon < 0.5 * max_loss) {
            for (size_t k = 0; k != loss.size(); ++k) {
                const double d = loss[k] / max_loss;
                pmf[k] = (float) (pow(beta, 1 - d) * pmf[k]);
            }
        } else {
            std::cout << "RESET" << std::endl;
            // reset to uniform distribution
            std::fill(pmf.begin(), pmf.end(), 1.0 / nrows);
        }
        normalize(pmf);
    }
}
