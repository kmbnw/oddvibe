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

    Booster::Booster(const size_t& seed) : m_seed(seed) {
    }

    void
    Booster::update_one(
        const DataSet& data,
        FloatVec& pmf,
        SizeVec& counts)
    const {
        const size_t nrows = data.nrows();

        EmpiricalSampler sampler(m_seed, pmf);

        const auto active = sampler.gen_samples(nrows);
        update_counts(active, counts);

        RTree tree;
        tree.fit(data, active);
        const auto yhats = tree.predict(data);
        auto loss = data.loss(yhats);
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

    std::pair<size_t, float>
    Booster::fit(const DataSet& data, size_t nrounds) const {
        const auto nrows = data.nrows();

        // set up initial uniform distribution over all instances
        FloatVec pmf(nrows, 1.0 / nrows);
        SizeVec counts(nrows, 0);

        for (size_t k = 0; k != nrounds; ++k) {
            update_one(data, pmf, counts);

            if (k == (nrounds - 2)) {
                for (size_t j = 0; j != nrows; ++j) {
                    const auto avg_count = 1.0 * counts[j] / (k + 1);
                    std::cout << std::setw(4) << std::left << j;
                    std::cout << std::fixed << std::setprecision(2);
                    std::cout << std::setw(7) << std::left << pmf[j];
                    std::cout << "avg_count[x] = " << std::setw(7) << std::left;
                    std::cout << avg_count << std::endl;
                }
            }
        }

        auto best = std::numeric_limits<float>::quiet_NaN();
        size_t best_idx = 1;

        const float actual_rounds = 1.0f * nrounds + 1;
        for (size_t j = 0; j != nrows; ++j) {
            const auto norm_count = 1.0 * counts[j] / actual_rounds;
            if (std::isnan(norm_count)) {
                throw std::logic_error("NaN for normalized count");
            }
            if (std::isnan(best) || norm_count > best) {
                best = norm_count;
                best_idx = j;
            }
        }
        return std::make_pair(best_idx, best);
    }
}
