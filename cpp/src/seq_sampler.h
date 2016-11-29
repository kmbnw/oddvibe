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

#ifndef KMBNW_ODVB_SEQ_SAMPLER_H
#define KMBNW_ODVB_SEQ_SAMPLER_H

namespace oddvibe {
    /**
     * Return sequential samples from start to end (inclusive).
     * This will loop around back to zero when it reaches the end.
     */
    class SequentialSampler: public Sampler {
        public:
            SequentialSampler(const size_t& start, const size_t& end);
            virtual size_t next_sample() override;
            virtual size_t size() override;

        private:
            const size_t m_start;
            const size_t m_end;
            size_t m_current = 0;
    };
}
#endif //KMBNW_ODVB_SEQ_SAMPLER_H
