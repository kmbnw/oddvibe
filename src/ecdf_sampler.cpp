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
    EmpiricalSampler::EmpiricalSampler(const size_t seed) :
       m_rand_engine(std::mt19937(seed)) {
    }

    SizeVec
    EmpiricalSampler::gen_samples(
            const size_t nrows,
            std::discrete_distribution<size_t>&& pmf) {

        SizeVec seq(nrows, 0);
        std::generate(
            seq.begin(),
            seq.end(),
            [&] { return pmf(m_rand_engine); });
        return seq;
    }
}
