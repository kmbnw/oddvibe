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

#include <utility>
#include <limits>
#include <stdexcept>
#include <cmath>
#include "rtree.h"

namespace oddvibe {
     RTree::RTree(
            const size_t ncols,
            const std::vector<float>& xs,
            const std::vector<float>& ys):
            m_nrows(ys.size()), m_ncols(ncols), m_xs(xs), m_ys(ys) {
        if (ys.size() != xs.size() / ncols) {
            throw std::invalid_argument("xs and ys must have same number of rows");
        }
    }

    float RTree::y_at(const size_t row_idx) const {
        if (row_idx < m_nrows) {
            return m_ys[row_idx];
        }
        throw std::out_of_range("row_idx out of range");
    }

    float RTree::x_at(const size_t row_idx, const size_t col_idx) const {
        if (row_idx >= m_nrows) {
            throw std::out_of_range("row_idx out of range");
        }
        if (col_idx >= m_ncols) {
            throw std::out_of_range("col_idx out of range");
        }

        return m_xs[(row_idx * m_ncols) + col_idx];
    }

    size_t RTree::nrows() const {
        return m_nrows;
    }

    size_t RTree::ncols() const {
        return m_ncols;
    }

    std::unordered_set<float>
    RTree::unique_values(
            const size_t col,
            const std::vector<bool>& active) const {
        std::unordered_set<float> uniques;

        for (size_t row = 0; row != m_nrows; ++row) {
            uniques.insert(m_xs[(row * m_ncols) + col]);
        }
        return uniques;
    }

    double RTree::calc_total_err(
            const size_t col,
            const float split,
            const std::vector<bool>& active,
            const float yhat_l,
            const float yhat_r) const {
        double err = 0;

        for (size_t row_j = 0; row_j != m_nrows; ++row_j) {
            auto x_j = m_xs[(row_j * m_ncols) + col];
            auto y_j = m_ys[row_j];
            auto yhat = x_j <= split ? yhat_l : yhat_r;

            err += pow((y_j - yhat), 2.0);
        }
        return err;
    }

    std::pair<float, float>
    RTree::calc_yhat(
            const size_t col,
            const float split,
            const std::vector<bool>& active) const {
        float yhat_l = 0;;
        float yhat_r = 0;
        size_t size_l = 0;
        size_t size_r = 0;

        for (size_t row_j = 0; row_j != m_nrows; ++row_j) {
            auto x_j = m_xs[(row_j * m_ncols) + col];
            auto y_j = m_ys[row_j];
            if (x_j <= split) {
                ++size_l;
                yhat_l = yhat_l + ((y_j - yhat_l) / size_l);
            } else {
                ++size_r;
                yhat_r = yhat_r + ((y_j - yhat_r) / size_r);
            }
        }
        return std::make_pair(yhat_l, yhat_r);
    }

    std::pair<size_t, float>
    RTree::best_split() const {
        std::vector<bool> active(m_ys.size(), true);
        return best_split(active);
    }

    std::pair<size_t, float>
    RTree::best_split(const std::vector<bool>& active) const {
        double best_err = std::numeric_limits<double>::quiet_NaN();
        size_t best_feature = -1;
        float best_split = std::numeric_limits<float>::quiet_NaN();
                        
        for (size_t col = 0; col != m_ncols; ++col) {
            auto uniques = unique_values(col, active);

            if (uniques.size() < 2) {
                continue;
            }

            for (auto const & split : uniques) {
                // calculate yhat for left and right side of split
                auto yhat = calc_yhat(col, split, active);
                auto yhat1 = yhat.first;
                auto yhat2 = yhat.second;

                // total squared error for left and right side of split
                auto err = calc_total_err(col, split, active, yhat1, yhat2);

                if (std::isnan(best_err) || err < best_err) {
                    best_err = err;
                    best_split = split;
                    best_feature = col;
                }
            }
        }
        return std::make_pair(best_feature, best_split);
    }
}
