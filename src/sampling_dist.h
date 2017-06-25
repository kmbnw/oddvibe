/*
 * Copyright 2017 Krysta M Bouzek
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

#ifndef KMBNW_ODVB_SAMPLING_DIST_H
#define KMBNW_ODVB_SAMPLING_DIST_H

namespace oddvibe {
    class SamplingDist {
        public:
            SamplingDist(const size_t nrows);

            SamplingDist(const FloatVec& pmf);

            void adjust_for_loss(const DoubleVec& loss);

            std::discrete_distribution<size_t> empirical_dist() const;

      private:
            FloatVec m_pmf;

            void reset();
    };
}
#endif //KMBNW_ODVB_SAMPLING_DIST_H
