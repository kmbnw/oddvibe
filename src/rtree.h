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
#include <future>
#include "split_point.h"

namespace oddvibe {

    /**
     * Regression decision tree
     */
    class RTree {
        public:
            class Trainer;

            RTree(RTree&& other) = default;
            RTree& operator=(RTree&& other) = default;

            RTree(const RTree& other) = delete;
            RTree& operator=(const RTree& other) = delete;

            ~RTree() = default;

            template <typename MatrixT>
            FloatVec predict(const MatrixT& xs) const {
                const auto nrows = xs.nrow();
                FloatVec yhats(nrows, floatNaN);
                SizeVec filter(nrows);
                std::iota(filter.begin(), filter.end(), 0);
                predict(xs, filter, yhats);

                return yhats;
            }

        private:
            float m_yhat = floatNaN;
            bool m_is_leaf = true;
            SplitPoint m_split;

            std::unique_ptr<RTree> m_left;
            std::unique_ptr<RTree> m_right;

            RTree(const float yhat) : m_yhat(yhat) { };

            RTree(
                    const float yhat,
                    const SplitPoint& split,
                    std::unique_ptr<RTree> left,
                    std::unique_ptr<RTree> right) :
                m_yhat(yhat),
                m_is_leaf(false),
                m_split(split),
                m_left(std::move(left)),
                m_right(std::move(right)) {

                if (!m_left) {
                    throw std::logic_error("Cannot have a null left child node");
                }
                if (!m_right) {
                    throw std::logic_error("Cannot have a null right child node");
                }

            }

            template <typename MatrixT>
            void
            predict(const MatrixT& xs, SizeVec& filter, FloatVec& yhat) const {
                if (m_is_leaf) {
                    for (const auto & row : filter) {
                        yhat[row] = m_yhat;
                    }
                } else {
                    const auto pivot = m_split.partition_idx(
                        xs, filter.begin(), filter.end());
                    SizeVec lsplit(filter.begin(), pivot);
                    SizeVec rsplit(pivot, filter.end());

                    m_left->predict(xs, lsplit, yhat);
                    m_right->predict(xs, rsplit, yhat);
                }
            }
    };

    class RTree::Trainer {
        public:
            Trainer(const size_t max_depth) : m_max_depth(max_depth) {}

            Trainer(Trainer&& other) = delete;
            Trainer& operator=(Trainer&& other) = delete;

            Trainer(const Trainer& other) = delete;
            Trainer& operator=(const Trainer& other) = delete;

            ~Trainer() = default;

            template <typename MatrixT, typename VectorT, typename FloatT>
            std::unique_ptr<RTree> fit(
                    const Dataset<MatrixT, VectorT, FloatT>& data,
                    const SizeVec& filter,
                    const size_t depth) const {
                if (filter.empty()) {
                    throw std::invalid_argument("Must have at least one entry");
                }

                const MatrixT& xs = data.xs();
                const VectorT& ys = data.ys();
                const auto yhat = mean(ys, filter.begin(), filter.end());
                if (std::isnan(yhat)) {
                    throw std::logic_error("Prediction is NaN");
                }

                bool force_leaf = (
                    depth >= m_max_depth ||
                    variance(ys, filter.begin(), filter.end()) < 1e-6);

                if (!force_leaf) {
                    const auto split = best_split(data, filter);

                    if (split.is_valid()) {
                        SizeVec part(filter);
                        const auto pivot = split.partition_idx(
                            xs, part.begin(), part.end());

                        const auto ndepth = depth + 1;
                        auto left = std::async(
                            std::launch::deferred,
                            [this, &data, &part, pivot, ndepth]() {
                                SizeVec lpart(part.begin(), pivot);
                                return fit(data, lpart, ndepth);
                            });
                        auto right = std::async(
                            std::launch::deferred,
                            [this, &data, &part, pivot, ndepth]() {
                                SizeVec rpart(pivot, part.end());
                                return fit(data, rpart, ndepth);
                            });

                        // stuck on C++ 11 b/c the Debian Rcpp build forces it?
                        // TODO I would like to fix that

                        auto ltree = left.get();
                        auto rtree = right.get();
                        return std::unique_ptr<RTree>(
                            new RTree(
                                yhat,
                                split,
                                std::move(ltree),
                                std::move(rtree)));
                    }
                }
                // leaf
                return std::unique_ptr<RTree>(new RTree(yhat));
            }
        private:
            size_t m_max_depth;
    };
}
#endif //KMBNW_ODVB_RTREE_H
