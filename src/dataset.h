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
#ifndef KMBNW_ODVB_DATASET_H
#define KMBNW_ODVB_DATASET_H

#include <cstddef>
#include <utility>

namespace oddvibe {
    template <typename MatrixT, typename VectorT>
    class Dataset {
        public:
            explicit Dataset<MatrixT, VectorT>(MatrixT& xs, VectorT& ys) {
                if (xs.nrow() != ys.size()) {
                    throw std::logic_error("X and Y row counts do not match");
                }
                m_xs = xs;
                m_ys = ys;
            }

            explicit Dataset<MatrixT, VectorT>(MatrixT&& xs, VectorT&& ys) {
                if (xs.nrow() != ys.size()) {
                    throw std::logic_error("X and Y row counts do not match");
                }
                m_xs = std::move(xs);
                m_ys = std::move(ys);
            }

            Dataset<MatrixT, VectorT>(Dataset<MatrixT, VectorT>&& other) = default;
            Dataset<MatrixT, VectorT>(const Dataset<MatrixT, VectorT>& other) = default;
            Dataset<MatrixT, VectorT>& operator=(const Dataset<MatrixT, VectorT>& other) = default;
            Dataset<MatrixT, VectorT>& operator=(Dataset<MatrixT, VectorT>&& other) = default;
            ~Dataset<MatrixT, VectorT>() = default;

            const MatrixT& xs() const { return m_xs; }
            const VectorT& ys() const { return m_ys; }

            FloatVec
            unique_x(const size_t col, const SizeVec& indices) const {
                std::unordered_set<float> uniques;

                for (const auto & row : indices) {
                    uniques.insert(m_xs(row, col));
                }
                return FloatVec(uniques.begin(), uniques.end());
            }

            size_t ncol() const {
                return m_xs.ncol();
            }

            size_t nrow() const {
                return m_xs.nrow();
            }

        private:
            MatrixT m_xs;
            VectorT m_ys;
    };
}
#endif //KMBNW_ODVB_DATASET_H
