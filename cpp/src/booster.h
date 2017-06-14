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
#include <unordered_map>
#include <functional>
#include "train_data.h"
#include "defs_x.h"

#ifndef KMBNW_ODVB_BOOSTER_H
#define KMBNW_ODVB_BOOSTER_H

namespace oddvibe {
    void
    update_counts(const SizeVec& src, SizeVec& counts);

    /**
     * Provides boosting capabilities to other models.
     */
    class Booster {
        public:
            Booster(const size_t& seed);

            Booster(const Booster& other) = delete;
            Booster& operator=(const Booster& other) = delete;

            std::pair<size_t, float>
            fit(const DataSet& data, const size_t nrows) const;

        private:
            size_t m_seed;

            void
            update_one(
                const DataSet& data,
                FloatVec& pmf,
                SizeVec& counts)
            const;
    };
}
#endif //KMBNW_ODVB_BOOSTER_H
