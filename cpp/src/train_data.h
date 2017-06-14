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
#include <unordered_set>
#include <utility>
#include "math_x.h"

#ifndef KMBNW_TRAIN_DATA_H
#define KMBNW_TRAIN_DATA_H

namespace oddvibe {
    /**
     * Abstraction over training data (features and response).
     * Immutable.
     */
    class DataSet {
        public:
            /**
             * Create new training data from flattened features and responses.
             * @param[in] ncols: Number of columns/features.
             * @param[in] xs: Flattened matrix of features: row0 followed by
             * row1, etc.
             * @param[in] ys: Response (independent variable).
             */
            DataSet(const size_t ncols, const FloatVec &xs, const FloatVec &ys);

            /**
             * No copy.
             */
            DataSet(const DataSet& other) = delete;
            /**
             * No copy.
             */
            DataSet& operator=(const DataSet& other) = delete;

            /**
             * Get the Y-value (response) at the given row.
             */
            float y_at(const size_t row) const;
            /**
             * Get the X-value (feature and value) at the given column and row.
             */
            float x_at(const size_t row, const size_t col) const;

            /**
             * Number of rows.
             */
            size_t nrows() const;

            /**
             * Number of columns (features).
             */
            size_t ncols() const;

            double mean_y(const SizeVec& row_idx) const;

            double variance_y(const SizeVec& row_idx) const;

            std::unordered_set<float>
            unique_x(const size_t col, const SizeVec& row_idx)
            const;

            /**
             * RMSE loss
             */
            DoubleVec loss(const FloatVec& yhat) const;

        private:
            size_t m_nrows;
            size_t m_ncols;
            FloatVec m_xs;
            FloatVec m_ys;

            size_t x_index(const size_t row, const size_t col) const;
    };
}
#endif //KMBNW_TRAIN_DATA_H
