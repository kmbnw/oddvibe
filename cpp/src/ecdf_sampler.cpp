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
#include <vector>
#include <random>
#include <ctime>
#include "ecdf_sampler.h"

namespace oddvibe {

    EmpiricalSampler::EmpiricalSampler(const size_t& seed, const std::vector<float>& pmf) :
            m_unif_dist(std::uniform_real_distribution<float>(0, 1)),
            m_rand_engine(std::mt19937(seed)) {

        fill_ecdf(pmf, m_ecdf);
    }

    size_t EmpiricalSampler::next_sample() {
        const float unif_val = m_unif_dist(m_rand_engine);

        const size_t last = m_ecdf.size() - 2;
        for (size_t k = 0; k < last; ++k) {
            if (unif_val > m_ecdf[k] && unif_val <= m_ecdf[k + 1]) {
                return k;
            }
        }
        return last;
    }

    void fill_ecdf(const std::vector<float>& pmf, std::vector<float>& ecdf) {
        ecdf.clear();

        float previous = 0;
        ecdf.push_back(previous);

        for (const auto& prob: pmf) {
            float next = prob + previous;
            ecdf.push_back(next);
            previous = next;
        }
    }
}
