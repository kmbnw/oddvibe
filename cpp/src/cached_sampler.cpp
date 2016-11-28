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

    CachedSampler::CachedSampler(Sampler& sampler, const size_t& nrows) {
        m_samples.clear();
        for (size_t k = 0; k < nrows; ++k) {
            m_samples.push_back(sampler.next_sample());
        }
    }

    size_t CachedSampler::next_sample() {
        size_t idx = m_current;
        m_current = (m_current + 1) % m_samples.size();
        return m_samples[idx];
    }

    void CachedSampler::add_counts(std::vector<unsigned int>& counts) {
        if (counts.size() != m_samples.size()) {
            throw std::logic_error("counts.size() != m_samples.size()");
        }

        for (const auto& sample_idx: m_samples) {
            counts[sample_idx]++;
        }
    }
}
