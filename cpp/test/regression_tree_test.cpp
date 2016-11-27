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
#include <unordered_map>
#include <cmath>
#include <vector>
#include <unordered_map>
#include "partitioner.h"
#include "regression_tree.h"
#include "regression_tree_test.h"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION(oddvibe::RegressionTreeTest);

namespace oddvibe {

    void RegressionTreeTest::setUp() {
    }

    void RegressionTreeTest::tearDown() {
    }

    // see PartitionerTest for the values predicted
    void RegressionTreeTest::test_predict() {
        // odd numbers are feature 1, even are feature 2
        const std::vector<float> xs {
            1.2f, 12.2f,
            3.4f, 2.6f,
            7.1f, 8.8f,
            5.2f, 8.8f
        };
        const std::vector<float> ys {
            8.0f,
            2.5f,
            0.0f,
            -36.2f
        };
        const size_t nfeatures = 2;
        const size_t depth = 1;

        Partitioner builder(nfeatures, depth, rmse, xs, ys);
        builder.build();
        const RegressionTree tree(builder);

        std::vector<float> yhats;
        tree.predict(xs, yhats);

        // split is done on feature 0, split value 3.4
        const float left_leaf = 5.25f;
        const float right_leaf = -18.1f;

        CPPUNIT_ASSERT_DOUBLES_EQUAL(left_leaf, yhats[0], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(left_leaf, yhats[1], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(right_leaf, yhats[2], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(right_leaf, yhats[3], m_tolerance);
    }

    void RegressionTreeTest::test_predict_depth2() {
        const std::vector<float> xs {
            3.15f, 8.19f,
            5.11f, 3.10f,
            3.61f, 6.14f,
            6.77f, 4.32f,
            5.93f, 6.01f,
            5.65f, 4.63f,
            6.36f, 6.02f,
            5.20f, 3.72f };

        const std::vector<float> ys {
            18.49f,
            18.02f,
            17.53f,
            20.57f,
            20.93f,
            14.59f,
            23.49f,
            23.30f
        };
        const std::vector<float> new_xs {
            2.15f, 8.19f,
            5.40f, 3.10f,
            5.7f, 5.9f,
            6.77f, 8.12f,
        };

        // top-level split on feature 0 at value 5.65
        // LHS second level split on feature 0 at value 5.20
        // RHS second level split on feature 1 at value 6.01
        // lr = left, then right, etc.
        const float ll_leaf = 19.335f;
        const float lr_leaf = 14.59f;
        const float rl_leaf = 20.75f;
        const float rr_leaf = 23.49f;

        const std::vector<float> expected {
            ll_leaf,
            lr_leaf,
            rl_leaf,
            rr_leaf
        };

        const size_t nfeatures = 2;
        const size_t depth = 2;

        Partitioner builder(nfeatures, depth, rmse, xs, ys);
        builder.build();

        const RegressionTree tree(builder);

        std::vector<float> yhats;
        tree.predict(new_xs, yhats);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[0], yhats[0], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[1], yhats[1], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[2], yhats[2], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[3], yhats[3], m_tolerance);
    }
}
