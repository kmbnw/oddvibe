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

#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include "../../src/rtree.h"
#include "../../src/float_matrix.h"
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

    void RTreeTest::test_best_split_none() {
        std::vector<float> xs {
            1.2f,  1.2f,  1.2f,  1.2f,
            12.2f, 12.2f, 12.2f, 12.2f
        };

        const std::vector<float> ys {
            8.0f,
            2.5f,
            8.0f,
            2.5f
        };
        const size_t nfeatures = 2;

        std::vector<size_t> seq(ys.size());
        std::iota(seq.begin(), seq.end(), 0);

        const Dataset<float> data(
            FloatMatrix<float>(nfeatures, xs),
            std::vector<float>(ys));

        const auto split = best_split(data, seq.begin(), seq.end());

        CPPUNIT_ASSERT_EQUAL(false, split.is_valid());
    }

    // perfect split on second feature; first is uninformative
    void RTreeTest::test_best_split_perfect() {
        const float split_val = 2.6f;

        std::vector<float> xs {
            1.2f,  1.2f,      1.2f,  1.2f,
            12.2f, split_val, 12.2f, split_val
        };

        const std::vector<float> ys {
            8.0f,
            2.5f,
            8.0f,
            2.5f
        };
        const size_t nfeatures = 2;

        std::vector<size_t> seq(ys.size());
        std::iota(seq.begin(), seq.end(), 0);

        const Dataset<float> data(
            FloatMatrix<float>(nfeatures, xs),
            std::vector<float>(ys));
        const auto split = best_split(data, seq.begin(), seq.end());
        const auto col = split.split_col();
        const auto value = split.split_val();

        size_t expected_feature = 1;

        CPPUNIT_ASSERT_EQUAL(true, split.is_valid());
        CPPUNIT_ASSERT_EQUAL(expected_feature, col);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(split_val, value, m_tolerance);
    }

    // perfect split on second feature; first is less informative
    void RTreeTest::test_best_split_near_perfect() {
        const float split_val = 2.6f;
        std::vector<float> xs {
            1.2f,  3.4f,      1.2f,  1.2f,
            12.2f, split_val, 12.2f, split_val
        };

        const std::vector<float> ys {
            8.0f,
            2.5f,
            8.0f,
            2.5f
        };
        const size_t nfeatures = 2;

        std::vector<size_t> seq(ys.size());
        std::iota(seq.begin(), seq.end(), 0);

        const Dataset<float> data(
            FloatMatrix<float>(nfeatures, xs),
            std::vector<float>(ys));
        const auto split = best_split(data, seq.begin(), seq.end());
        const auto col = split.split_col();
        const auto value = split.split_val();

        const size_t expected_feature = 1;

        CPPUNIT_ASSERT_EQUAL(true, split.is_valid());
        CPPUNIT_ASSERT_EQUAL(expected_feature, col);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(split_val, value, m_tolerance);
    }

    // 3rd feature provides best split
    void RTreeTest::test_best_split_formula() {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> dist(0.0f, 10.0f);
        const size_t nrolls = 1000;

        std::vector<float> xs1(nrolls);
        std::vector<float> xs2(nrolls);
        std::vector<float> xs3(nrolls);

        std::vector<float> ys(nrolls);

        const float beta1 = 0;
        const float beta2 = 2.5;
        const float beta3 = 35;
        const float intercept = 1.2;

        for (size_t j = 0; j != nrolls; ++j) {
            auto x1 = dist(generator);
            auto x2 = dist(generator);
            auto x3 = dist(generator);
            xs1[j] = x1;
            xs2[j] = x2;
            xs3[j] = x3;
            ys[j] = intercept + beta1 * x1 + beta2 * x2 + beta3 * x3;
        }

        // lay out the resulting flattened matrix
        std::vector<float> xs;
        std::copy(xs1.begin(), xs1.end(), std::back_inserter(xs));
        std::copy(xs2.begin(), xs2.end(), std::back_inserter(xs));
        std::copy(xs3.begin(), xs3.end(), std::back_inserter(xs));

        CPPUNIT_ASSERT_EQUAL(nrolls * 3, xs.size());

        const size_t nfeatures = 3;

        std::vector<size_t> seq(ys.size());
        std::iota(seq.begin(), seq.end(), 0);

        const Dataset<float> data(
            FloatMatrix<float>(nfeatures, xs),
            std::vector<float>(ys));
        const auto split = best_split(data, seq.begin(), seq.end());
        const auto col = split.split_col();
        const auto value = split.split_val();

        size_t expected_feature = 2;
        float expected_val = 4.885; // TODO hand-calc this

        CPPUNIT_ASSERT_EQUAL(true, split.is_valid());
        CPPUNIT_ASSERT_EQUAL(expected_feature, col);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected_val, value, m_tolerance);
    }
}
