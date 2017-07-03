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


#ifndef KMBNW_ODVB_FLOAT_MATRIX_H
#define KMBNW_ODVB_FLOAT_MATRIX_H

namespace oddvibe {
    // follow R's NumericVector API where necessary to facilitate easier use
    template <typename FloatT>
    class FloatMatrix {
        public:
            FloatMatrix() = default;

            /**
             * Move construct new instance from flat matrix.
             * @param[in] ncols: Number of columns/features.
             * @param[in] xs: Flattened matrix of features: row0 followed by
             * row1, etc.
             */
            explicit FloatMatrix(const size_t ncols, std::vector<FloatT>&& xs) {
                if (xs.empty()) {
                    if (ncols > 0) {
                        throw std::invalid_argument(
                            "Cannot set ncols for empty vector");
                    }
                } else {
                    if (xs.size() % ncols != 0) {
                        throw std::invalid_argument(
                            "Invalid shape for input vector");
                    }
                    m_ncols = ncols;
                    m_nrows = xs.size() / ncols;
                    m_xs = std::move(xs);
                }
            }

            /**
             * Copy construct new instance from flat matrix.
             * @param[in] ncols: Number of columns/features.
             * @param[in] xs: Flattened matrix of features: row0 followed by
             * row1, etc.
             */
            explicit FloatMatrix(const size_t ncols, const std::vector<FloatT>& xs) {
                if (xs.empty()) {
                    if (ncols > 0) {
                        throw std::invalid_argument(
                            "Cannot set ncols for empty vector");
                    }
                } else {
                    if (xs.size() % ncols != 0) {
                        throw std::invalid_argument(
                            "Invalid shape for input vector");
                    }
                    m_ncols = ncols;
                    m_nrows = xs.size() / ncols;
                    m_xs = xs;
                }
            }

            FloatMatrix(FloatMatrix&& other) = default;
            FloatMatrix& operator=(FloatMatrix&& other) = default;

            FloatMatrix(const FloatMatrix& other) = default;
            FloatMatrix& operator=(const FloatMatrix& other) = default;

            ~FloatMatrix() = default;

            FloatT operator() (const size_t row, const size_t col) const {
                return m_xs[x_index(row, col)];
            }

            /**
             * Number of rows.
             */
            size_t nrow() const {
                return m_nrows;
            }

            /**
             * Number of columns (features).
             */
            size_t ncol() const {
                return m_ncols;
            }

            private:
                size_t m_nrows = 0;
                size_t m_ncols = 0;
                std::vector<FloatT> m_xs;

                size_t x_index(const size_t row, const size_t col) const {
                    //        return (row * m_ncols) + col;
                    return (col * m_nrows) + row;
                }
    };
}
#endif //KMBNW_ODVB_FLOAT_MATRIX_H
