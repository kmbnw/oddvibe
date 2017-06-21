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
#include "float_matrix.h"
#include "dataset.h"

#ifndef KMBNW_ODVB_BOOSTER_H
#define KMBNW_ODVB_BOOSTER_H

namespace oddvibe {
    void update_counts(const SizeVec& src, SizeVec& counts);

    /**
     * Provides boosting capabilities to other models.
     */
    class Booster {
        public:
            Booster(const size_t &seed);

            Booster(const Booster &other) = delete;
            Booster &operator=(const Booster &other) = delete;


            template<typename MatrixT, typename VectorT>
            FloatVec
            fit(Dataset<MatrixT, VectorT> dataset, const size_t nrounds) const {
                const auto nrows = dataset.nrows();

                // set up initial uniform distribution over all instances
                FloatVec pmf(nrows, 1.0 / nrows);
                SizeVec counts(nrows, 0);

                for (size_t k = 0; k != nrounds; ++k) {
                    update_one(dataset, pmf, counts);

                    /*if (k == (nrounds - 2)) {
                        for (size_t j = 0; j != nrows; ++j) {
                            const auto avg_count = 1.0 * counts[j] / (k + 1);
                            std::cout << std::setw(4) << std::left << j;
                            std::cout << std::fixed << std::setprecision(2);
                            std::cout << std::setw(7) << std::left << pmf[j];
                            std::cout << "avg_count[x] = " << std::setw(7) << std::left;
                            std::cout << avg_count << std::endl;
                        }
                    }*/
                }

                return normalize_counts(counts, nrounds);
            }

      private:
            size_t m_seed;

            void update_one(
                const Dataset<FloatMatrix, FloatVec>& dataset,
                FloatVec &pmf,
                SizeVec &counts)
            const;
    };
}
#endif //KMBNW_ODVB_BOOSTER_H
