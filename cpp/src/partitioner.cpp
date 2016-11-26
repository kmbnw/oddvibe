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

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <functional>
#include "partitioner.h"

namespace oddvibe {
    Partitioner::Partitioner(
            size_t ncols,
            size_t depth,
            const std::function<double(const std::vector<float>&, const std::vector<float>&)> &err_fn,
            const std::vector<float> &xs,
            const std::vector<float> &ys): m_ncols(ncols), m_xs(xs), m_ys(ys), m_err_fn(err_fn) {
        const size_t tree_sz = pow(2, depth);

        m_feature_idxs.clear();
        m_feature_idxs.resize(tree_sz, 0);
        m_split_vals.clear();
        m_split_vals.resize(tree_sz, 0);
    }

    void Partitioner::build() {
        const std::vector<bool> row_filter(m_ys.size(), true);

        build(1, row_filter);
    }

    void Partitioner::build(const size_t &node_idx, const std::vector<bool> &row_filter) {
        if (node_idx >= m_feature_idxs.size()) {
            m_predictions[node_idx] = filtered_mean(m_ys, row_filter);
            return;
        }

        // init in the case of one element
        size_t feature_idx = 0;
        float split_value = nan("");
        float error = nan("");

        std::vector<float> left;
        std::vector<float> right;
        size_t xs_len = m_xs.size();

        // for each feature
        for (size_t col_idx = 0; col_idx != m_ncols; ++col_idx) {

            // for each split point candidate of the feature
            for (size_t i = 0; i != xs_len; i += m_ncols) {
                const size_t filter_idx = i / m_ncols;

                if (!row_filter[filter_idx]) {
                    continue;
                }

                // look at the parent node to see if we already split
                // on this value
                float x = m_xs[i + col_idx];
                if (node_idx > 1 &&
                    col_idx == m_feature_idxs[node_idx / 2] &&
                    x == m_split_vals[node_idx / 2])
                {
                    continue;
                }

                left.clear();
                right.clear();

                float y = m_ys[i / m_ncols];
                left.push_back(y);

                // divide the data and calc error
                for (size_t k = 0; k != xs_len; k += m_ncols) {
                    const size_t filter_idx_k = k / m_ncols;
                    if (k == i || !row_filter[filter_idx_k]) {
                        continue;
                    }

                    float other_x = m_xs[k + col_idx];
                    float other_y = m_ys[k / m_ncols];

                    if (other_x <= x) {
                        left.push_back(other_y);
                    } else {
                        right.push_back(other_y);
                    }
                }
                float err = m_err_fn(left, right);

                if (isnan(error) || err < error) {
                    error = err;
                    split_value = x;
                    feature_idx = col_idx;
                }
            }
        }

        m_feature_idxs[node_idx] = feature_idx;
        m_split_vals[node_idx] = split_value;

        const size_t left_idx = node_idx * 2;
        const size_t right_idx = node_idx * 2 + 1;

        // left
        std::vector<bool> tmp_filter(row_filter.size(), true);
        set_row_filter(row_filter, tmp_filter, feature_idx, split_value, true);

        build(left_idx, tmp_filter);

        //right
        std::fill(tmp_filter.begin(), tmp_filter.end(), true);
        set_row_filter(row_filter, tmp_filter, feature_idx, split_value, false);

        build(right_idx, tmp_filter);
    }

    void Partitioner::set_row_filter(
            const std::vector<bool> &row_filter,
            std::vector<bool> &tmp_filter,
            size_t feature_idx,
            const float &split_value,
            bool left) const {

        for (size_t row_idx = 0; row_idx < m_xs.size(); row_idx += m_ncols) {
            const size_t filter_idx = row_idx / m_ncols;
            const float x = m_xs[row_idx + feature_idx];

            if (left) {
                tmp_filter[filter_idx] = row_filter[filter_idx] && (x <= split_value);
            } else {
                tmp_filter[filter_idx] = row_filter[filter_idx] && (x > split_value);
            }

        }
    }

    double rmse(const std::vector<float> &left, const std::vector<float> &right) {
        std::vector<float> ldiff(left.size());
        std::vector<float> rdiff(right.size());

        float left_mean = std::accumulate(left.begin(), left.end(), 0.0f) / left.size();
        float right_mean = std::accumulate(right.begin(), right.end(), 0.0f) / right.size();

        std::transform(left.begin(), left.end(), ldiff.begin(),
            [left_mean](float x) { return pow(x - left_mean, 2); });

        std::transform(right.begin(), right.end(), rdiff.begin(),
            [right_mean](float x) { return pow(x - right_mean, 2); });

        float err = sqrt(std::accumulate(ldiff.begin(), ldiff.end(), 0.0f) +
                         std::accumulate(rdiff.begin(), rdiff.end(), 0.0f));

        return err;
    }

    float filtered_mean(const std::vector<float> &ys, const std::vector<bool> &row_filter) {
        double sum = 0;
        size_t count = 0;
        for (size_t i = 0; i != ys.size(); ++i) {
            if (row_filter[i]) {
                sum += ys[i];
                ++count;
            }
        }

        // predict mean
        return (float) (sum / count);
    }
}
