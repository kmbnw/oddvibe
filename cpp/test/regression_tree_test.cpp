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

        CPPUNIT_ASSERT_DOUBLES_EQUAL(left_leaf, yhats[0], _tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(left_leaf, yhats[1], _tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(right_leaf, yhats[2], _tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(right_leaf, yhats[3], _tolerance);

//        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.4f, split_val, _tolerance);
    }

    void RegressionTreeTest::test_predict_depth2() {

        // via hand-calculation
        // 5.20, feature 0 for LHS split
        // 6.01, feature 1 for LHS split
//        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.65f, split_vals[1], _tolerance);
  //      CPPUNIT_ASSERT_DOUBLES_EQUAL(5.20f, split_vals[2], _tolerance);
    //    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.01f, split_vals[3], _tolerance);
    }
}
