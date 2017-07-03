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

            /**
             * Predict for an input feature matrix.
             *
             * \param xs The feature matrix to generate predictions for.
             * \return A vector of predictions, one for each row of the input
             * matrix.
             */
            std::vector<FloatT> predict(const FloatMatrix<FloatT>& xs) const {
                const auto nrows = xs.nrow();
                constexpr auto nan_val = std::numeric_limits<FloatT>::quiet_NaN();
                std::vector<FloatT> yhats(nrows, nan_val);
                std::vector<size_t> filter(nrows);
                std::iota(filter.begin(), filter.end(), 0);
                predict(xs, filter.begin(), filter.end(), yhats);

                return yhats;
            }

        private:
            FloatT m_yhat = std::numeric_limits<FloatT>::quiet_NaN();
            bool m_is_leaf = true;
            SplitPoint<FloatT> m_split;

            std::unique_ptr<RTree<FloatT>> m_left;
            std::unique_ptr<RTree<FloatT>> m_right;

            /**
             * Create an RTree leaf node.
             *
             * \param yhat The predictive value at this leaf.
             */
            RTree<FloatT>(const FloatT yhat) : m_yhat(yhat) { };

            /**
             * Create an RTree interior node.
             *
             * \param yhat The predictive value at this interior node.
             * \param left The left branch of this RTree.
             * \param right The right branch of this RTree.
             */
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

            /**
             * Recursively predict for a filtered input feature matrix.
             *
             * The elements from the range `[first, last]` are used to filter
             * the input data; they are row indices into the Dataset xs() and
             * ys() values that will be used for prediction at each level of
             * the tree.  On each left/right branch of an interior node, the row
             * indexes are further filtered based on the node's SplitPoint.
             *
             * \param xs The feature matrix to generate predictions for.
             * \param first BidirectionalIterator to the initial position of
             * the row indexes.
             * \param last BidirectionalIterator to the final position of
             * the row indexes.
             * \param An accumulator vector of predictions, one for each row of
             * xs.  This will be filled in by the successive recursive calls.
             */
            template <typename BidirectionalIterator>
            void predict(
                    const FloatMatrix<FloatT>& xs,
                    BidirectionalIterator first,
                    BidirectionalIterator last,
                    std::vector<FloatT>& yhat) const {
                if (m_is_leaf) {
                    for (auto row = first; row != last; row = std::next(row)) {
                        yhat[*row] = m_yhat;
                    }
                } else {
                    const auto pivot = m_split.partition_idx(xs, first, last);
                    m_left->predict(xs, first, pivot, yhat);
                    m_right->predict(xs, pivot, last, yhat);
                }
            }
    };

    template <typename FloatT>
    class RTree<FloatT>::Trainer {
        public:
            /**
             * Create a new RTree Trainer with a given max depth
             *
             * \param max_depth The max depth/height of the fitted tree.
             */
            Trainer(const size_t max_depth) : m_max_depth(max_depth) {}

            Trainer(Trainer&& other) = delete;
            Trainer& operator=(Trainer&& other) = delete;

            Trainer(const Trainer& other) = delete;
            Trainer& operator=(const Trainer& other) = delete;

            ~Trainer() = default;

            /**
             * Fit an RTree to filtered data.
             *
             * Fitting the model is done by considering all possible
             * (feature, value) pairs as split points for each level of the
             * RTree.  This function calls itself recursively on the left and
             * right branches of the RTree whenever it finds a valid SplitPoint
             * (i.e. when there is nonzero variance in the response values,
             * when there is more than one unique element for at least one
             * feature, and when depth has not exceeded the max depth of this
             * Trainer).
             *
             * The elements from the range `[first, last]` are used to filter
             * the input data; they are row indices into the Dataset xs() and
             * ys() values that will be used for fitting.  On each left/right
             * interior node, the row indexes are further filtered based on the
             * chosen SplitPoint.
             *
             * \param first BidirectionalIterator to the initial position of
             * the row indexes.
             * \param last BidirectionalIterator to the final position of
             * the row indexes.
             * \param depth The tree height at which the resulting RTree node
             * resides (used to limit tree height).  Each left/right call of
             * fit() will have its depth incremented by one.
             * \return A pointer to the RTree (node) at this level; the very
             * first call of fit() will return a pointer to the root of the tree.
             */
            template <typename BidirectionalIterator>
            std::unique_ptr<RTree<FloatT>> fit(
                    const Dataset<FloatT>& data,
                    const BidirectionalIterator first,
                    const BidirectionalIterator last,
                    const size_t depth) const {
                if (first == last) {
                    throw std::invalid_argument("Must have at least one entry");
                }

                const FloatMatrix<FloatT>& xs = data.xs();
                const std::vector<FloatT>& ys = data.ys();
                const auto yhat = mean<FloatT>(ys, first, last);
                if (std::isnan(yhat)) {
                    throw std::logic_error("Prediction is NaN");
                }

                bool force_leaf = (
                    depth >= m_max_depth ||
                    variance<FloatT>(ys, first, last) < 1e-6);

                if (!force_leaf) {
                    const auto split = best_split(data, first, last);

                    if (split.is_valid()) {
                        const auto pivot = split.partition_idx(xs, first, last);

                        const auto ndepth = depth + 1;
                        auto left = std::async(
                            std::launch::deferred,
                            [this, &data, first, last, pivot, ndepth]() {
                                return fit(data, first, pivot, ndepth);
                            });
                        auto right = std::async(
                            std::launch::deferred,
                            [this, &data, first, last, pivot, ndepth]() {
                                return fit(data, pivot, last, ndepth);
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
