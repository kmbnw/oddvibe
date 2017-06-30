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
#include "ecdf_sampler.h"
#include "rtree.h"
#include "sampling_dist.h"

#ifndef KMBNW_ODVB_BOOSTER_H
#define KMBNW_ODVB_BOOSTER_H

namespace oddvibe {
    void update_counts(const SizeVec& indexes, SizeVec& counts) {
        for (const auto & idx : indexes) {
            ++counts[idx];
        }
    }

    /**
     * Provides boosting capabilities to other models.
     */
    class Booster {
        public:
            Booster(const size_t &seed) : m_seed(seed) {}

            Booster(const Booster &other) = delete;
            Booster &operator=(const Booster &other) = delete;


            template <typename MatrixT, typename VectorT>
            FloatVec
            fit(const MatrixT& xs, const VectorT& ys, const size_t nrounds) const {
                const auto nrows = xs.nrow();

                // set up initial uniform distribution over all instances
                SamplingDist pmf(nrows);
                SizeVec counts(nrows, 0);
                EmpiricalSampler sampler(m_seed);

                for (size_t k = 0; k != nrounds; ++k) {
                    const auto active = sampler.gen_samples(nrows, pmf);
                    update_counts(active, counts);

                    RTree tree(xs, ys, active, 0, 6);
                    const auto loss = loss_seq(ys, tree.predict(xs));

                    pmf.adjust_for_loss(loss);
                }

                return normalize_counts(counts, nrounds);
            }

      private:
            size_t m_seed;
    };
}
#endif //KMBNW_ODVB_BOOSTER_H
