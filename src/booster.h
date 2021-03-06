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

/*! \file */

namespace oddvibe {
    /**
     * Provides boosting capabilities to RTree models.
     * \sa RTree
     */
    class Booster {
        public:
            /**
             * Create a new instance with the specified random seed.
             *
             * \param seed Random seed to initialize with.
             */
            Booster(const size_t &seed) : m_seed(seed) {}

            Booster(const Booster &other) = delete;
            Booster &operator=(const Booster &other) = delete;

            /**
             * Find possible outliers using boosted RTrees
             *
             * \param data Dataset of feature matrix and response vector to fit.
             * \param nrounds Number of rounds of boosting (often called
             * number of trees).
             * \return A vector of normalized counts, one for each row of the
             * input data.  Each element represents the number of times that
             * row of data was chosen during boosting, normalized by `nrounds`.
             */
            template <typename FloatT>
            std::vector<float> fit_counts(
                    const Dataset<FloatT>& data,
                    const size_t nrounds) const {
                const FloatMatrix<FloatT>& xs = data.xs();
                const std::vector<FloatT>& ys = data.ys();
                const auto nrows = data.nrow();

                // set up initial uniform distribution over all instances
                SamplingDist pmf(nrows);
                std::vector<size_t> counts(nrows, 0);
                EmpiricalSampler sampler(m_seed);

                const typename RTree<FloatT>::Trainer trainer(6);

                for (size_t k = 0; k != nrounds; ++k) {
                    auto active = sampler.gen_samples(nrows, pmf);

                    for (const auto & idx : active) {
                        ++counts[idx];
                    }

                    const auto tree = trainer.fit(
                        data, active.begin(), active.end(), 0);
                    const auto loss = loss_seq(ys, tree->predict(xs));

                    pmf.adjust_for_loss(loss);
                }

                return divide_vector(counts, nrounds);
            }

      private:
            size_t m_seed;
    };
}
#endif //KMBNW_ODVB_BOOSTER_H
