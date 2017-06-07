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

#include "rtree.h"
#include <limits>
#include <stdexcept>
#include <cmath>

namespace oddvibe {
    RTree::RTree(const DataSet& data, const std::vector<bool>& active) : m_active(active) {
        //auto split = best_split(data);
        // TODO left child, right child
    }

    RTree::RTree(const DataSet& data) : m_active(std::vector<bool>(true, data.nrows())) {
        //auto split = best_split(data);
        // TODO left child, right child
    }

    std::unordered_set<float>
    RTree::unique_values(
            const DataSet& data,
            const size_t col) const {
        std::unordered_set<float> uniques;
        const auto nrows = data.nrows();

        for (size_t row = 0; row != nrows; ++row) {
            uniques.insert(data.x_at(row, col));
        }
        return uniques;
    }

    double RTree::calc_total_err(
            const DataSet& data,
            const size_t col,
            const float split,
            const float yhat_l,
            const float yhat_r) const {
        double err = std::numeric_limits<double>::quiet_NaN();
        const auto nrows = data.nrows();
        bool init = false;

        for (size_t row_j = 0; row_j != nrows; ++row_j) {
            if (m_active[row_j]) {
                auto x_j = data.x_at(row_j, col);
                auto y_j = data.y_at(row_j);
                auto yhat = x_j <= split ? yhat_l : yhat_r;

                if (!init) {
                    init = true;
                    err = 0;
                }
                err += pow((y_j - yhat), 2.0);
            }
        }
        return err;
    }

    std::pair<float, float>
    RTree::calc_yhat(
            const DataSet& data,
            const size_t col,
            const float split) const {
        float yhat_l = 0;;
        float yhat_r = 0;
        size_t size_l = 0;
        size_t size_r = 0;
        const auto nrows = data.nrows();

        for (size_t row_j = 0; row_j != nrows; ++row_j) {
            auto x_j = data.x_at(row_j, col);
            auto y_j = data.y_at(row_j);
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

    SplitData
    RTree::best_split(const DataSet& data) const {
        double best_err = std::numeric_limits<double>::quiet_NaN();
        size_t best_feature = -1;
        float best_value = std::numeric_limits<float>::quiet_NaN();
        bool init = false;

        const auto ncols = data.ncols();
        for (size_t col = 0; col != ncols; ++col) {
            auto uniques = unique_values(data, col);

            if (uniques.size() < 2) {
                continue;
            }

            // TODO check for zero variance

            for (const auto & split : uniques) {
                // calculate yhat for left and right side of split
                const auto yhat = calc_yhat(data, col, split);
                const auto yhat_l = yhat.first;
                const auto yhat_r = yhat.second;

                if (std::isnan(yhat_l) || std::isnan(yhat_r)) {
                    continue;
                }

                // total squared error for left and right side of split
                const auto err = calc_total_err(data, col, split, yhat_l, yhat_r);

                // TODO randomly allow the same error as best to 'win'
                if (!init || (!std::isnan(err) && err < best_err)) {
                    init = true;
                    best_err = err;
                    best_value = split;
                    best_feature = col;
                }
            }
        }
        return SplitData(best_value, best_feature, best_err);
    }
}
