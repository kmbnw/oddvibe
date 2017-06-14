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
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include "algorithm_x.h"
#include "ecdf_sampler.h"

namespace oddvibe {
    EmpiricalSampler::EmpiricalSampler(
            const size_t seed,
            const std::vector<float>& pmf) :
       m_size(pmf.size()),
       m_rand_engine(std::mt19937(seed)),
       m_dist(std::discrete_distribution<size_t>(pmf.begin(), pmf.end())) {
    }

    size_t EmpiricalSampler::next_sample() {
        return m_dist(m_rand_engine);
    }

    size_t EmpiricalSampler::size() {
        return m_size;
    }

    SizeVec EmpiricalSampler::gen_samples(const size_t nrows) {
        SizeVec seq(nrows, 0);
        std::generate(
            seq.begin(),
            seq.end(),
            [&] { return this->next_sample(); });
        return seq;
    }
}
