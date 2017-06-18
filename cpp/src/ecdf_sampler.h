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
#include <vector>
#include <random>
#include "defs_x.h"

#ifndef KMBNW_ODVB_ECDF_SAMPLER_H
#define KMBNW_ODVB_ECDF_SAMPLER_H

namespace oddvibe {
    class EmpiricalSampler {
        public:
            // pmf == probability mass function
            EmpiricalSampler(const size_t seed, const std::vector<float>& pmf);

            size_t next_sample();

            SizeVec gen_samples(const size_t nrows);

        private:
            std::mt19937 m_rand_engine;
            std::discrete_distribution<size_t> m_dist;
    };
}
#endif //KMBNW_ODVB_ECDF_SAMPLER_H
