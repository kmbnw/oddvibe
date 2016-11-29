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
#include <unordered_map>
#include <functional>
#include "partitioner.h"

#ifndef KMBNW_ODVB_BOOSTER_H
#define KMBNW_ODVB_BOOSTER_H

namespace oddvibe {
    /**
     * Normalize a vector to sum to 1 (e.g. proper probability mass function).
     * @param[inout] pmf The vector to normalize; overwritten in-place.
     */
    void normalize(std::vector<float>& pmf);

    /**
     * Add 1 to each index of counts for active sample indexes.
     * Does so by repeatedly calling sampler.next_sample().
     * E.g. if this Sampler holds [1, 10, 10, 11] as its indexes,
     * this is equivalent to counts[1]++; counts[10]++;
     * counts[10]++; counts[11]++;
     *
     * @param[in] sampler The sampler to get sample indexes from.
     * @param[inout] counts A sample index count accumulator.  Must be the
     * same size as Sampler.nrows.
     * from.
     */
    void add_counts(Sampler& sampler, std::vector<unsigned int>& counts);

    /**
     * Provides boosting capabilities to other models.
     */
    class Booster {
        public:
            Booster(
                Partitioner* const builder,
                const std::function<double(const std::vector<float>&, const std::vector<float>&)> &err_fn);

            Booster(const Booster& other) = delete;
            Booster& operator=(const Booster& other) = delete;

            void update_one(
                std::vector<float>& pmf,
                std::vector<unsigned int>& counts,
                const std::vector<float> &xs,
                const std::vector<float> &ys) const;

        private:
            Partitioner* const m_builder;
            const size_t m_seed;
            const std::function<double(const std::vector<float>&, const std::vector<float>&)> m_err_fn;
    };
}
#endif //KMBNW_ODVB_BOOSTER_H
