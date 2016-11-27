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
    void normalize(std::vector<float>& pmf);

    /**
     * Provides boosting capabilities to other models.
     */
    class Booster {
        public:
            Booster(
                Partitioner& builder,
                const std::function<double(const std::vector<float>&, const std::vector<float>&)> &err_fn);

            Booster(const Booster& other) = delete;
            Booster& operator=(const Booster& other) = delete;

            /**
             * Run one round of the boosting algorithm.
             */
            void update_one(const std::vector<float> &xs, const std::vector<float> &ys) const;

        private:
            Partitioner* const m_builder;
            const size_t m_seed;
            const std::function<double(const std::vector<float>&, const std::vector<float>&)> m_err_fn;
    };
}
#endif //KMBNW_ODVB_BOOSTER_H
