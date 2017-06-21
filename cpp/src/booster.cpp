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
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <functional>
#include "booster.h"
#include "rtree.h"
#include "ecdf_sampler.h"
#include "math_x.h"

namespace oddvibe {
    void
    update_counts(const SizeVec& src, SizeVec& counts) {
        for (const auto & idx : src) {
            ++counts[idx];
        }
    }

    Booster::Booster(const size_t &seed) : m_seed(seed)
    { }

    void Booster::update_one(
        const Dataset<FloatMatrix, FloatVec>& dataset,
        FloatVec& pmf,
        SizeVec& counts)
    const {
        const size_t nrows = dataset.nrows();

        EmpiricalSampler sampler(m_seed, pmf);

        const auto active = sampler.gen_samples(nrows);
        update_counts(active, counts);

        RTree tree;
        tree.fit(dataset, active);
        const auto yhats = tree.predict(dataset.xs());

        auto loss = loss_seq(dataset.ys(), yhats);
        const double max_loss = *std::max_element(loss.begin(), loss.end());

        double epsilon = 0.0;
        for (size_t k = 0; k != loss.size(); ++k) {
            epsilon += pmf[k] * loss[k];
        }

        const double beta = epsilon / (max_loss - epsilon);

        if (epsilon < 0.5 * max_loss) {
            std::transform(
                pmf.begin(),
                pmf.end(),
                loss.begin(),
                pmf.begin(),
                [beta = beta, max_loss = max_loss](float pmf_k, double loss_k) {
                    return (float) (pow(beta, 1 - loss_k / max_loss) * pmf_k);
                });
        } else {
            std::cout << "RESET" << std::endl;
            // reset to uniform distribution
            std::fill(pmf.begin(), pmf.end(), 1.0 / nrows);
        }
        normalize(pmf);
    }
}
