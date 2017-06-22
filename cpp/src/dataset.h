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
#include <stdexcept>
#include "defs_x.h"
#include "math_x.h"

#ifndef KMBNW_ODVB_DATASET_H
#define KMBNW_ODVB_DATASET_H

namespace oddvibe {
    template<typename MatrixT, typename VectorT>
    class Dataset {
        public:
            Dataset(const MatrixT& xs, const VectorT& ys) {
                if (xs.nrows() != ys.size()) {
                    throw std::invalid_argument("Mismatched number of rows");
                }
                m_xs = xs;
                m_ys = ys;
            }

            Dataset(MatrixT&& xs, VectorT&& ys) {
                if (xs.nrows() != ys.size()) {
                    throw std::invalid_argument("Mismatched number of rows");
                }
                m_xs = std::move(xs);
                m_ys = std::move(ys);
            }

            // TODO explicit defaults

            size_t nrows() const {
                return m_xs.nrows();
            }

            size_t ncols() const {
                return m_xs.ncols();
            }

            const MatrixT& xs() const {
                return m_xs;
            }

            const VectorT& ys() const {
                return m_ys;
            }

        private:
            MatrixT m_xs;
            VectorT m_ys;
    };
}
#endif //KMBNW_ODVB_DATASET_H
