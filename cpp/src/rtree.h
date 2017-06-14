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

#ifndef KMBNW_RTREE_H
#define KMBNW_RTREE_H

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include "split_data.h"
#include "train_data.h"

namespace oddvibe {

    /**
     * Regression decision tree
     */
    class RTree {
        public:
            class Fitter;
            RTree(const Fitter& fitter);

            FloatVec predict(const DataSet& data) const;

        private:
            float m_yhat = std::numeric_limits<double>::quiet_NaN();
            bool m_is_leaf = true;
            size_t m_split_col = 0;
            float m_split_val = std::numeric_limits<double>::quiet_NaN();
            std::unique_ptr<RTree> m_left;
            std::unique_ptr<RTree> m_right;

            void validate();

            void predict(
                const DataSet& data,
                const std::vector<bool>& active,
                FloatVec& predicted)
            const;

            void fill_active(
                const DataSet& data,
                const std::vector<bool>& init_active,
                std::vector<bool>& left_active,
                std::vector<bool>& right_active)
            const;
    };

    /**
     * Fit data for a regression decision tree
     */
    class RTree::Fitter {
        friend class RTree;

        public:
            Fitter(const SizeVec& active_idx);

            /**
             * No copy.
             */
            Fitter(const Fitter& other) = delete;
            /**
             * No copy.
             */
            Fitter& operator=(const Fitter& other) = delete;

            SplitData best_split(const DataSet& data) const;

            RTree build();

            void fit(const DataSet& data);

        private:
            float m_yhat = std::numeric_limits<double>::quiet_NaN();
            bool m_is_leaf = true;
            size_t m_split_col = 0;
            float m_split_val = std::numeric_limits<double>::quiet_NaN();
            SizeVec m_active_idx;
            std::unique_ptr<Fitter> m_left;
            std::unique_ptr<Fitter> m_right;

            double calc_total_err(
                const DataSet& data,
                const size_t split_col,
                const float split_val,
                const float yhat_l,
                const float yhat_r)
            const;

            std::pair<float, float> fit_children(
                const DataSet& data,
                const size_t split_col,
                const float split_val)
            const;

            void fill_row_idx(
                const DataSet& data,
                const size_t split_col,
                const float split_val,
                SizeVec& left_rows,
                SizeVec& right_rows)
            const;
    };
}
#endif //KMBNW_RTREE_H
