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

#ifndef KMBNW_ODVB_SAMPLING_DIST_H
#define KMBNW_ODVB_SAMPLING_DIST_H

namespace oddvibe {
    /**
     * Empirical distribution used to sample N rows of data during boosting.
     * \sa Booster
     */
    class SamplingDist {
        public:
            /**
             * Create a new instance with a uniform distribution.
             */
            SamplingDist(const size_t nrows);

            /**
             * Create a new instance with the given distribution.
             * \param The distribution to initialize this instance with.
             */
            SamplingDist(const std::vector<float>& pmf);

            /**
             * Update the underlying distribution of this instance using
             * the loss vector from a round of boosting.
             * \loss The loss for each row of input data.  This must be the
             * same size() as the distribution being updated, and will throw
             * an exception if it is not.
             * \sa Booster::fit
             */
            void adjust_for_loss(const std::vector<double>& loss);

            /**
             * \return A copy of the discrete empirical distribution underlying
             * this instance.
             */
            std::discrete_distribution<size_t> empirical_dist() const;

      private:
            size_t m_size;
            std::vector<float> m_pmf;

            /**
             * Return this instance to a uniform distribution.
             */
            void reset();
    };
}
#endif //KMBNW_ODVB_SAMPLING_DIST_H
