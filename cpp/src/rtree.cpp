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

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stack>
#include "rtree.h"

namespace oddvibe {

    RTree::RTree(const float yhat, const bool is_leaf, const SplitData& split) :
        m_yhat(yhat),
        m_is_leaf(is_leaf),
        m_split(split) {
    }

    void RTree::validate() {
        std::stack<RTree*> stk;
        stk.push(this);

        RTree* p;
        while(!stk.empty()) {
            p = stk.top();
            stk.pop();
            if (!(p->m_is_leaf)) {
                if (!(p->m_left)) {
                    throw std::logic_error("validate: left child is NULL");
                }
                if (!(p->m_left)) {
                    throw std::logic_error("validate: right child is NULL");
                }
                stk.push(p->m_left.get());
                stk.push(p->m_right.get());
            }
        }
    }

    void
    RTree::predict(
            const DataSet& data,
            const BoolVec& active,
            FloatVec& yhat)
    const {
        const auto nrows = data.nrows();
        if (m_is_leaf) {
            for (size_t row = 0; row != nrows; ++row) {
                if (active[row]) {
                    yhat[row] = m_yhat;
                }
            }
        } else {
            BoolVec left_active(nrows, false);
            BoolVec right_active(nrows, false);
            fill_active(data, active, left_active, right_active);

            m_left->predict(data, left_active, yhat);
            m_right->predict(data, right_active, yhat);
        }
    }

    FloatVec RTree::predict(const DataSet& data) const {
        const auto nan = std::numeric_limits<double>::quiet_NaN();
        const auto nrows = data.nrows();
        FloatVec yhats(nrows, nan);
        BoolVec active(nrows, true);

        predict(data, active, yhats);

        return yhats;
    }

    void
    RTree::fill_active(
            const DataSet& data,
            const BoolVec& init_active,
            BoolVec& left_active,
            BoolVec& right_active)
    const {
        const auto nrows = data.nrows();
        const auto split_col = m_split.split_col();
        const auto split_val = m_split.split_val();

        for (size_t row = 0; row != nrows; ++row) {
            if (init_active[row]) {
                auto x_j = data.x_at(row, split_col);
                if (x_j <= split_val) {
                    left_active[row] = true;
                } else {
                    right_active[row] = true;
                }
            }
        }
    }

    RTree::Fitter::Fitter(const SizeVec& active_idx) {
        if (active_idx.empty()) {
            throw std::invalid_argument("Must have at least one active row");
        }

        m_active_idx = active_idx;
    }

    void
    RTree::Fitter::fit(const DataSet& data) {
        m_yhat = data.mean_y(m_active_idx);

        if (std::isnan(m_yhat)) {
            throw std::logic_error("Prediction cannot be NaN");
        }

        if (data.variance_y(m_active_idx) > 1e-6) {
            auto split = best_split(data);

            if (split.is_valid()) {
                m_is_leaf = false;
                m_split = split;

                SizeVec left_idx;
                SizeVec right_idx;
                fill_row_idx(data, m_split, left_idx, right_idx);

                m_left = std::make_unique<Fitter>(left_idx);
                m_right = std::make_unique<Fitter>(right_idx);

                m_left->fit(data);
                m_right->fit(data);
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

    RTree RTree::Fitter::build() {
        std::stack<RTree::Fitter*> fit_stk;
        fit_stk.push(this);

        RTree root(this->m_yhat, this->m_is_leaf, this->m_split);
        std::stack<RTree*> tree_stk;
        tree_stk.push(&root);

        RTree::Fitter* fit;
        RTree* tree;
        while (!fit_stk.empty()) {
            fit = fit_stk.top();
            fit_stk.pop();

            tree = tree_stk.top();
            tree_stk.pop();

            if (!(fit->m_is_leaf)) {
                // create the basic data parts of the child nodes but not
                // their children
                const auto left = fit->m_left.get();
                const auto right = fit->m_right.get();
                tree->m_left = std::make_unique<RTree>(
                    left->m_yhat,
                    left->m_is_leaf,
                    left->m_split);
                tree->m_right = std::make_unique<RTree>(
                    right->m_yhat,
                    right->m_is_leaf,
                    right->m_split);

                fit_stk.push(fit->m_left.get());
                fit_stk.push(fit->m_right.get());

                tree_stk.push(tree->m_left.get());
                tree_stk.push(tree->m_right.get());
            }
        }
        root.validate();
        return root;
    }

    void
    RTree::Fitter::fill_row_idx(
            const DataSet& data,
            const SplitData& split,
            SizeVec& left_rows,
            SizeVec& right_rows)
    const {
        const auto split_col = split.split_col();
        const auto split_val = split.split_val();

        for (const auto & row : m_active_idx) {
            const auto x = data.x_at(row, split_col);
            if (x <= split_val) {
                left_rows.push_back(row);
            } else {
                right_rows.push_back(row);
            }
        }
    }

    double
    RTree::Fitter::calc_total_err(
            const DataSet& data,
            const SplitData& split,
            const float yhat_l,
            const float yhat_r)
    const {
        const auto split_col = split.split_col();
        const auto split_val = split.split_val();

        double err = 0;
        for (const auto & row : m_active_idx) {
            auto x_j = data.x_at(row, split_col);
            auto y_j = data.y_at(row);
            auto yhat = x_j <= split_val ? yhat_l : yhat_r;
            err += pow((y_j - yhat), 2.0);
        }
        return err;
    }

    std::pair<float, float>
    RTree::Fitter::fit_children(const DataSet& data, const SplitData& split) const {
        SizeVec left_idx;
        SizeVec right_idx;
        fill_row_idx(data, split, left_idx, right_idx);
        const float yhat_l = data.mean_y(left_idx);
        const float yhat_r = data.mean_y(right_idx);

        return std::make_pair(yhat_l, yhat_r);
    }

    SplitData
    RTree::Fitter::best_split(const DataSet& data)
    const {
        SplitData best;
        double best_err = std::numeric_limits<double>::quiet_NaN();
        bool init = false;

        const auto ncols = data.ncols();
        for (size_t split_col = 0; split_col != ncols; ++split_col) {
            auto uniques = data.unique_x(split_col, m_active_idx);

            if (uniques.size() < 2) {
                continue;
            }

            for (const auto & split_val : uniques) {
                // calculate yhat for left and right side of split_val
                SplitData split(split_val, split_col);
                const auto yhat = fit_children(data, split);
                const auto yhat_l = yhat.first;
                const auto yhat_r = yhat.second;

                if (std::isnan(yhat_l) || std::isnan(yhat_r)) {
                    continue;
                }

                // total squared error for left and right side of split_val
                const auto err = calc_total_err(data, split, yhat_l, yhat_r);

                // TODO randomly allow the same error as best to 'win'
                if (!init || (!std::isnan(err) && err < best_err)) {
                    init = true;
                    best = split;
                    best_err = err;
                }
            }
        }
        return best;
    }
}
