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

#ifndef KMBNW_ODVB_RTREE_H
#define KMBNW_ODVB_RTREE_H

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include "split_point.h"
#include "algorithm_x.h"
#include "dataset.h"

namespace oddvibe {

    /**
     * Regression decision tree
     */
    class RTree {
        public:
            RTree(RTree&& other) = default;
            RTree& operator=(RTree&& other) = default;

            RTree(const RTree& other) = delete;
            RTree& operator=(const RTree& other) = delete;

            ~RTree() = default;

            template<typename MatrixT, typename VectorT>
            RTree(const Dataset<MatrixT, VectorT>& data, const SizeVec& rows) {
                if (rows.empty()) {
                    throw std::invalid_argument("Must have at least one entry");
                }

                // defensive copy
                SizeVec filter(rows);

                m_yhat = mean(data.ys(), filter.begin(), filter.end());
                if (std::isnan(m_yhat)) {
                    throw std::logic_error("Prediction cannot be NaN");
                }

                SplitPoint split;
                if (variance(data.ys(), filter.begin(), filter.end()) > 1e-6) {
                    split = best_split(data.xs(), data.ys(), filter);

                    if (split.is_valid()) {
                        m_split = split;
                        m_is_leaf = false;

                        const auto pivot = m_split.partition_idx(data.xs(), filter);
                        SizeVec lsplit(filter.begin(), pivot);
                        SizeVec rsplit(pivot, filter.end());

                        m_left = std::make_unique<RTree>(data, lsplit);
                        m_right = std::make_unique<RTree>(data, rsplit);
                    }
                }

                if (!m_is_leaf) {
                    if (!m_left) {
                        throw std::logic_error("Cannot have a null left child node");
                    }
                    if (!m_right) {
                        throw std::logic_error("Cannot have a null right child node");
                    }
                }
            }

            template<typename MatrixT>
            FloatVec predict(const MatrixT& mat) const {
                const auto nrows = mat.nrows();
                FloatVec yhats(nrows, floatNaN);
                auto filter = sequential_ints(nrows);

                predict(mat, filter, yhats);

                return yhats;
            }

        private:
            float m_yhat = floatNaN;
            bool m_is_leaf = true;
            SplitPoint m_split;

            std::unique_ptr<RTree> m_left;
            std::unique_ptr<RTree> m_right;

            template<typename MatrixT>
            void
            predict(const MatrixT& mat, SizeVec& filter, FloatVec& yhat) const {
                if (m_is_leaf) {
                    for (const auto & row : filter) {
                        yhat[row] = m_yhat;
                    }
                } else {
                    const auto pivot = m_split.partition_idx(mat, filter);
                    SizeVec lsplit(filter.begin(), pivot);
                    SizeVec rsplit(pivot, filter.end());

                    m_left->predict(mat, lsplit, yhat);
                    m_right->predict(mat, rsplit, yhat);
                }
            }
    };
}
#endif //KMBNW_ODVB_RTREE_H
