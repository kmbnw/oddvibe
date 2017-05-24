/*
 * Copyright 2016 Krysta M Bouzek
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
#include <stdexcept>
#include <cmath>
#include <unordered_map>
#include "train_data.h"

namespace oddvibe {
     TrainingData::TrainingData(
            const size_t ncols,
            const std::vector<float>& xs,
            const std::vector<float>& ys):
            m_nrows(ys.size()), m_ncols(ncols), m_xs(xs), m_ys(ys) {
        if (ys.size() != xs.size() / ncols) {
            throw std::invalid_argument("xs and ys must have same number of rows");
        }
    }

    float TrainingData::y_at(const size_t row_idx) const {
        if (row_idx < m_nrows) {
            return m_ys[row_idx];
        }
        throw std::out_of_range("row_idx out of range");
    }

    float TrainingData::x_at(const size_t row_idx, const size_t col_idx) const {
        if (row_idx >= m_nrows) {
            throw std::out_of_range("row_idx out of range");
        }
        if (col_idx >= m_ncols) {
            throw std::out_of_range("col_idx out of range");
        }

        return m_xs[(row_idx * m_ncols) + col_idx];
    }

    size_t TrainingData::nrows() const {
        return m_nrows;
    }

    size_t TrainingData::ncols() const {
        return m_ncols;
    }

    float TrainingData::filtered_mean(const std::vector<bool> &row_filter) const {
        double sum = 0;
        size_t count = 0;
        for (size_t i = 0; i != m_ys.size(); ++i) {
            if (row_filter[i]) {
                sum += m_ys[i];
                ++count;
            }
        }

        return (float) (sum / count);
    }

    double TrainingData::best_split() const {
        double best_err = nan();
        size_t best_feature = -1;
        double best_split = nan();
        float yhat_l = 0;
        size_t size_l = 0;
        float yhat_r = 0;
        size_t size_r = 0;
                        
        for (auto col = 0; col != m_ncols; ++col) {
            unordered_map<float, bool> uniques;

            for (auto row = 0; row != m_nrows; ++row) {
                uniques[m_xs[(row * m_ncols) + col]] = true;
            }

            for (auto const & kv : uniques) {
                auto current_split = kv.first;

                // calculate yhat for left and right side of split
                for (auto row_j = 0; row_j != m_nrows; ++row_j) {
                    auto x_j = m_xs[(row_j * m_ncols) + col];
                    auto y_j = m_ys[row_j];
                    if (x_j <= current_split) {
                        yhat_l = yhat_l + ((y_j - yhat_l) / size_l);
                    } else {
                        yhat_r = yhat_r + ((y_j - yhat_r) / size_r);
                    }
                }

                // calculate total squared error for left and right side of split
                double err = 0;
                for (auto row_j = 0; row_j != m_nrows; ++row_j) {
                    auto x_j = m_xs[(row_j * m_ncols) + col];
                    auto y_j = m_ys[row_j];
                    auto yhat = x_j <= current_split ? yhat_l : yhat_r;

                    err += (y_j - yhat) ^ 2;
                }
                if (isnan(best_err) || err < best_err) {
                    best_err = err;
                    best_split = current_split;
                    best_feature = col;
                }
            }
        }
    }
}
