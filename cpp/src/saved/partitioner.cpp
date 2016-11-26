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
#include <algorithm>
#include <cmath>
#include "partitioner.h"

namespace oddvibe {
    Partitioner::Partitioner(
            const size_t &ncols,
            const std::function<double(const std::vector<float>&, const std::vector<float>&)> &err_fn,
            const std::vector<float> &xs,
            const std::vector<float> &ys) {
        _xs = xs;
        _ys = ys;
        _ncols = ncols;
        _err_fn = err_fn;
    }

    void Partitioner::find_splits(
            const size_t &depth,
            std::vector<size_t> &feature_idxs,
            std::vector<float> &split_vals) const {
        std::vector<bool> row_filter(_ys.size(), true);

        find_splits(depth, row_filter, feature_idxs, split_vals);
    }

    void Partitioner::find_splits(
            const size_t &depth,
            std::vector<bool> &row_filter,
            std::vector<size_t> &feature_idxs,
            std::vector<float> &split_vals) const {
        // init in the case of one element
        size_t feature_idx = 0;
        float split_value = nan("");
        float error = nan("");

        std::vector<float> left;
        std::vector<float> right;
        size_t xs_len = _xs.size();

        // for each feature
        for (size_t col_idx = 0; col_idx != _ncols; ++col_idx) {

            // for each split point candidate of the feature
            for (size_t i = 0; i != xs_len; i += _ncols) {
                if (!row_filter[i]) {
                    continue;
                }

                left.clear();
                right.clear();

                float x = _xs[i + col_idx];
                float y = _ys[i / _ncols];
                left.push_back(y);

                // divide the data and calc error
                for (size_t k = 0; k != xs_len; k += _ncols) {
                    if (k == i || !row_filter[k]) {
                        continue;
                    }

                    float other_x = _xs[k + col_idx];
                    float other_y = _ys[k / _ncols];

                    if (other_x <= x) {
                        left.push_back(other_y);
                    } else {
                        right.push_back(other_y);
                    }
                }
                float err = _err_fn(left, right);

                if (isnan(error) || err < error) {
                    std::cout << "Err: " << err << " idx: " << col_idx << std::endl;
                    error = err;
                    split_value = x;
                    feature_idx = col_idx;
                }
            }
        }
        feature_idxs.push_back(feature_idx);
        split_vals.push_back(split_value);

        if (depth > 1) {
            // left
            std::cout << "Split left: " << split_value << std::endl;
            set_row_filter(row_filter, feature_idx, split_value, true);
            find_splits(depth - 1, row_filter, feature_idxs, split_vals);

            //right
            std::cout << "Split right: " << split_value << std::endl;
            set_row_filter(row_filter, feature_idx, split_value, false);
            find_splits(depth - 1, row_filter, feature_idxs, split_vals);
        }
    }

    void Partitioner::set_row_filter(
            std::vector<bool> &row_filter,
            const int &feature_idx,
            const float &split_value,
            bool left) const {

        for (size_t row_idx = 0; row_idx < _xs.size(); row_idx += _ncols) {
            row_filter[row_idx] = _xs[row_idx + feature_idx] <= split_value;
            if (!left) {
                row_filter[row_idx] = !row_filter[row_idx];
            }
        }
    }

    double split_error(const std::vector<float> &left, const std::vector<float> &right) {
        std::vector<float> ldiff(left.size());
        std::vector<float> rdiff(right.size());

        float left_mean = std::accumulate(left.begin(), left.end(), 0.0f) / left.size();
        float right_mean = std::accumulate(right.begin(), right.end(), 0.0f) / right.size();

        std::transform(left.begin(), left.end(), ldiff.begin(),
            [left_mean](float x) { return pow(x - left_mean, 2); });

        std::transform(right.begin(), right.end(), rdiff.begin(),
            [right_mean](float x) { return pow(x - right_mean, 2); });

        float err = std::accumulate(ldiff.begin(), ldiff.end(), 0.0f) +
                    std::accumulate(rdiff.begin(), rdiff.end(), 0.0f);

        return err;
    }
}
