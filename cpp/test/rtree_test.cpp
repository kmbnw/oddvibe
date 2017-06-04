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
#include "rtree.h"
#include "rtree_test.h"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION(oddvibe::RTreeTest);

namespace oddvibe {

    void RTreeTest::setUp() {
    }

    void RTreeTest::tearDown() {
    }

    // perfect split on second feature; first is uninformative
    void RTreeTest::test_best_split_perfect() {
        // odd numbers are feature 1, even are feature 2
        const std::vector<float> xs {
            1.2f, 12.2f,
            1.2f, 2.6f,
            1.2f, 12.2f,
            1.2f, 2.6f
        };
        const std::vector<float> ys {
            8.0f,
            2.5f,
            8.0f,
            2.5f
        };
        const size_t nfeatures = 2;

        const RTree tree(nfeatures, xs, ys);
        auto split = tree.best_split();
        auto col = split.first;
        auto value = split.second;

        size_t expected_feature = 1;

        CPPUNIT_ASSERT_EQUAL(expected_feature, col);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.6f, value, m_tolerance);
    }
}
