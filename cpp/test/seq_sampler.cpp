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
#include "seq_sampler.h"

namespace oddvibe {

    SequentialSampler::SequentialSampler(const size_t& start, const size_t& end) :
            m_start(start), m_end(end), m_current(start) {
    }

    size_t SequentialSampler::next_sample() {
        size_t idx = m_current;
        m_current = (m_current + 1) % m_end;
        return idx;
    }

    size_t SequentialSampler::size() {
        return m_end - m_start;
    }
}