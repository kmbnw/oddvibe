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

#include <algorithm>
#include "sampling_dist.h"
#include "math_x.h"

namespace oddvibe {
    SamplingDist::SamplingDist(const size_t nrows) {
        m_pmf = FloatVec(nrows, 1.0 / nrows);
    }

    SamplingDist::SamplingDist(const FloatVec& pmf) {
        // TODO check for sum to 1
        m_pmf = pmf;
    }

    void SamplingDist::reset() {
        std::fill(m_pmf.begin(), m_pmf.end(), 1.0 / m_pmf.size());
    }

    void SamplingDist::adjust_for_loss(const DoubleVec& loss) {
        const double max_loss = *std::max_element(loss.begin(), loss.end());

        double epsilon = 0.0;
        for (size_t k = 0; k != loss.size(); ++k) {
            epsilon += m_pmf[k] * loss[k];
        }

        const double beta = epsilon / (max_loss - epsilon);

        if (epsilon < 0.5 * max_loss) {
            std::transform(
                m_pmf.begin(),
                m_pmf.end(),
                loss.begin(),
                m_pmf.begin(),
                [beta = beta, max_loss = max_loss](float pmf_k, double loss_k) {
                    return (float) (pow(beta, 1 - loss_k / max_loss) * pmf_k);
                });
        }  else {
            //std::cout << "RESET" << std::endl;
            // reset to uniform distribution
            reset();
        }
        normalize(m_pmf);
    }

    std::discrete_distribution<size_t>
    SamplingDist::empirical_dist() const {
        return std::discrete_distribution<size_t>(m_pmf.begin(), m_pmf.end());
    }

}
