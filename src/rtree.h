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
    template <typename FloatT>
    class RTree {
        public:
            class Trainer;

            RTree<FloatT>(RTree<FloatT>&& other) = default;
            RTree<FloatT>& operator=(RTree<FloatT>&& other) = default;

            RTree<FloatT>(const RTree<FloatT>& other) = delete;
            RTree<FloatT>& operator=(const RTree<FloatT>& other) = delete;

            ~RTree<FloatT>() = default;

            std::vector<FloatT> predict(const FloatMatrix<FloatT>& xs) const {
                const auto nrows = xs.nrow();
                constexpr auto nan_val = std::numeric_limits<FloatT>::quiet_NaN();
                std::vector<FloatT> yhats(nrows, nan_val);
                std::vector<size_t> filter(nrows);
                std::iota(filter.begin(), filter.end(), 0);
                predict(xs, filter, yhats);

                return yhats;
            }

        private:
            FloatT m_yhat = std::numeric_limits<FloatT>::quiet_NaN();
            bool m_is_leaf = true;
            SplitPoint<FloatT> m_split;

            std::unique_ptr<RTree<FloatT>> m_left;
            std::unique_ptr<RTree<FloatT>> m_right;

            RTree<FloatT>(const FloatT yhat) : m_yhat(yhat) { };

            RTree<FloatT>(
                    const FloatT yhat,
                    const SplitPoint<FloatT>& split,
                    std::unique_ptr<RTree<FloatT>> left,
                    std::unique_ptr<RTree<FloatT>> right) :
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

            void predict(
                    const FloatMatrix<FloatT>& xs,
                    std::vector<size_t>& filter,
                    std::vector<FloatT>& yhat) const {
                if (m_is_leaf) {
                    for (const auto & row : filter) {
                        yhat[row] = m_yhat;
                    }
                } else {
                    const auto pivot = m_split.partition_idx(
                        xs, filter.begin(), filter.end());
                    std::vector<size_t> lsplit(filter.begin(), pivot);
                    std::vector<size_t> rsplit(pivot, filter.end());

                    m_left->predict(xs, lsplit, yhat);
                    m_right->predict(xs, rsplit, yhat);
                }
            }
    };

    template <typename FloatT>
    class RTree<FloatT>::Trainer {
        public:
            Trainer(const size_t max_depth) : m_max_depth(max_depth) {}

            Trainer(Trainer&& other) = delete;
            Trainer& operator=(Trainer&& other) = delete;

            Trainer(const Trainer& other) = delete;
            Trainer& operator=(const Trainer& other) = delete;

            ~Trainer() = default;

            std::unique_ptr<RTree<FloatT>> fit(
                    const Dataset<FloatT>& data,
                    const std::vector<size_t>& filter,
                    const size_t depth) const {
                if (filter.empty()) {
                    throw std::invalid_argument("Must have at least one entry");
                }

                const FloatMatrix<FloatT>& xs = data.xs();
                const std::vector<FloatT>& ys = data.ys();
                const auto yhat = mean<FloatT>(ys, filter.begin(), filter.end());
                if (std::isnan(yhat)) {
                    throw std::logic_error("Prediction is NaN");
                }

                bool force_leaf = (
                    depth >= m_max_depth ||
                    variance<FloatT>(ys, filter.begin(), filter.end()) < 1e-6);

                if (!force_leaf) {
                    const auto split = best_split(data, filter);

                    if (split.is_valid()) {
                        std::vector<size_t> part(filter);
                        const auto pivot = split.partition_idx(
                            xs, part.begin(), part.end());

                        const auto ndepth = depth + 1;
                        auto left = std::async(
                            std::launch::deferred,
                            [this, &data, &part, pivot, ndepth]() {
                                std::vector<size_t> lpart(part.begin(), pivot);
                                return fit(data, lpart, ndepth);
                            });
                        auto right = std::async(
                            std::launch::deferred,
                            [this, &data, &part, pivot, ndepth]() {
                                std::vector<size_t> rpart(pivot, part.end());
                                return fit(data, rpart, ndepth);
                            });

                        // stuck on C++ 11 b/c the Debian Rcpp build forces it?
                        // TODO I would like to fix that

                        auto ltree = left.get();
                        auto rtree = right.get();
                        return std::unique_ptr<RTree<FloatT>>(
                            new RTree<FloatT>(
                                yhat,
                                split,
                                std::move(ltree),
                                std::move(rtree)));
                    }
                }
                // leaf
                return std::unique_ptr<RTree<FloatT>>(new RTree<FloatT>(yhat));
            }
        private:
            size_t m_max_depth;
    };
}
#endif //KMBNW_ODVB_RTREE_H
