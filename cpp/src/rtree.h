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
#include "float_matrix.h"
#include "dataset.h"

namespace oddvibe {

    /**
     * Regression decision tree
     */
    class RTree {
        public:
            RTree() = default;
            RTree(RTree&& other) = default;
            RTree& operator=(RTree&& other) = default;

            RTree(const RTree& other) = delete;
            RTree& operator=(const RTree& other) = delete;

            ~RTree() = default;

            FloatVec predict(const FloatMatrix& mat) const;

            void fit(
                const Dataset<FloatMatrix, FloatVec>& dataset,
                const SizeVec& filter);

            SplitPoint best_split(
                const Dataset<FloatMatrix, FloatVec>& dataset,
                const SizeVec& filter) const;

        private:
            float m_yhat = floatNaN;
            bool m_is_leaf = true;
            SplitPoint m_split;

            std::unique_ptr<RTree> m_left;
            std::unique_ptr<RTree> m_right;

            void fit(
                const Dataset<FloatMatrix, FloatVec>& dataset,
                SizeVec& filter);

            void predict(
                const FloatMatrix& mat,
                SizeVec& filter,
                FloatVec& yhat) const;
    };
}
#endif //KMBNW_ODVB_RTREE_H
