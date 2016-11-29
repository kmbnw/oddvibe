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
#include <vector>
#include <random>
#include "sampler.h"

#ifndef KMBNW_ODVB_ECDF_SAMPLER_H
#define KMBNW_ODVB_ECDF_SAMPLER_H

namespace oddvibe {
    class EmpiricalSampler: public Sampler {
        public:
            // pmf == probability mass function
            EmpiricalSampler(const size_t& seed, const std::vector<float>& pmf);
            virtual size_t next_sample() override;
            virtual size_t size() override;

        private:
            const size_t m_size;
            std::uniform_real_distribution<float> m_unif_dist;
            std::mt19937 m_rand_engine;
            // empirical CDF
            std::vector<float> m_ecdf;
    };

    void fill_ecdf(const std::vector<float>& pmf, std::vector<float>& ecdf);
}
#endif //KMBNW_ODVB_ECDF_SAMPLER_H
