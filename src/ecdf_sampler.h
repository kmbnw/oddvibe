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
#include "sampling_dist.h"

#ifndef KMBNW_ODVB_ECDF_SAMPLER_H
#define KMBNW_ODVB_ECDF_SAMPLER_H

namespace oddvibe {
    /**
     * Generate samples of row indexes from a given distribution.
     */
    class EmpiricalSampler {
        public:
            /**
             * Create a new instance with the specified random seed.
             *
             * \param seed Random seed to initialize with.
             */
            EmpiricalSampler(const size_t seed);

            /**
             * Generate empirical samples with replacement from a given
             * distribution.
             *
             * A sample is conceptually a row index into a feature matrix.
             *
             * \param nrows The number of samples to generate.
             * \param pmf The empirical distribution to generate row indexes
             * from.
             * \return A vector of randomly sampled row indexes, each within the
             * range of `[0, pmf.size())`.
             */
            std::vector<size_t>
            gen_samples(const size_t nrows, const SamplingDist& pmf);

        private:
            std::mt19937 m_rand_engine;
    };
}
#endif //KMBNW_ODVB_ECDF_SAMPLER_H
