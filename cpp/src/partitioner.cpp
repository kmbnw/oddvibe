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
            const size_t& ncols,
            const size_t& depth,
            const std::vector<float> &xs,
            const std::vector<float> &ys,
            const std::function<double(const std::vector<float>&, const std::vector<float>&)> &err_fn):
            m_ncols(ncols), m_tree_sz(pow(2, depth)), m_xs(xs), m_ys(ys), m_err_fn(err_fn) {
        if (ys.size() != xs.size() / ncols) {
            throw std::invalid_argument("xs and ys do not have the same number of instance rows");
        }
    }

    void Partitioner::reset() {
        m_feature_idxs.clear();
        m_feature_idxs.resize(m_tree_sz, 0);
        m_split_vals.clear();
        m_split_vals.resize(m_tree_sz, 0);
    }

    void Partitioner::build(Sampler& sampler) {
        const std::vector<bool> row_filter(m_ys.size(), true);

        reset();
        build(sampler, 1, row_filter);
    }

    /**
     * Build internal data structures for constructing a decision tree.
     *
     * @param[in] node_idx The index into the tree array for the current level.
     * 2 * K for left child, 2 * K + 1 for right child, with the root at 1 (not zero).
     *
     * @param[in] row_filter A mask for skipping input rows.  Used for parent node
     * splits to 'deactivate' a row when its feature and split value are not valid
     * for the current branch.  A true element means 'include this row'.
     */
    void Partitioner::build(
            Sampler& sampler,
            const size_t &node_idx,
            const std::vector<bool> &row_filter) {
        if (node_idx >= m_feature_idxs.size()) {
            m_predictions[node_idx] = filtered_mean(m_ys, row_filter);
            return;
        }

        const size_t nrows = m_ys.size();

        // init in the case of one element
        size_t feature_idx = 0;
        float split_value = nan("");
        float error = nan("");

        std::vector<float> left;
        std::vector<float> right;

        // for each feature
        for (size_t col_idx = 0; col_idx != m_ncols; ++col_idx) {

            // for each split point candidate of the feature
            size_t row_idx = 0;
            size_t count = 0;

            while (count++ < nrows) {
                row_idx = sampler.next_sample();

                if (!row_filter[row_idx]) {
                    continue;
                }

                // look at the parent node to see if we already split
                // on this value
                float x = m_xs[(row_idx * m_ncols) + col_idx];
                if (node_idx > 1 &&
                    col_idx == m_feature_idxs[node_idx / 2] &&
                    x == m_split_vals[node_idx / 2])
                {
                    continue;
                }

                left.clear();
                right.clear();

                float y = m_ys[row_idx];
                left.push_back(y);

                // divide the data and calc error
                for (size_t row_idx_k = 0; row_idx_k != nrows; ++row_idx_k) {
                    if (row_idx_k == row_idx || !row_filter[row_idx_k]) {
                        continue;
                    }

                    float other_x = m_xs[(row_idx_k * m_ncols) + col_idx];
                    float other_y = m_ys[row_idx_k];

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

        build(sampler, left_idx, tmp_filter);

        //right
        std::fill(tmp_filter.begin(), tmp_filter.end(), true);
        set_row_filter(row_filter, tmp_filter, feature_idx, split_value, false);

        build(sampler, right_idx, tmp_filter);
    }

    void Partitioner::set_row_filter(
            const std::vector<bool> &row_filter,
            std::vector<bool> &tmp_filter,
            size_t feature_idx,
            const float &split_value,
            bool left) const {

        for(size_t row_idx = 0; row_idx != m_ys.size(); ++row_idx) {
            const float x = m_xs[(row_idx * m_ncols) + feature_idx];

            if (left) {
                tmp_filter[row_idx] = row_filter[row_idx] && (x <= split_value);
            } else {
                tmp_filter[row_idx] = row_filter[row_idx] && (x > split_value);
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
