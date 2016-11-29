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

#include <stdexcept>
#include "cached_sampler.h"

namespace oddvibe {

    CachedSampler::CachedSampler(Sampler& sampler) {
        m_samples.clear();
        for (size_t k = 0; k != sampler.size(); ++k) {
            m_samples.push_back(sampler.next_sample());
        }
    }

    size_t CachedSampler::next_sample() {
        size_t idx = m_current;
        m_current = (m_current + 1) % m_samples.size();
        return m_samples[idx];
    }

    size_t CachedSampler::size() {
        return m_samples.size();
    }
}
