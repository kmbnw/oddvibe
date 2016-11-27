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
#include "seq_sampler.h"
#include "partitioner_test.h"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION(oddvibe::PartitionerTest);

namespace oddvibe {

    void PartitionerTest::setUp() {
    }

    void PartitionerTest::tearDown() {
    }

    void PartitionerTest::test_find_split() {
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

        SequentialSampler sampler(0, ys.size());
        Partitioner builder(nfeatures, depth, rmse, xs, ys);
        builder.build(sampler);

        size_t feature_idx = builder.m_feature_idxs[1];
        float split_val = builder.m_split_vals[1];

        CPPUNIT_ASSERT(0 == feature_idx);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.4f, split_val, _tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.25f, builder.m_predictions.find(2)->second, _tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-18.1f, builder.m_predictions.find(3)->second, _tolerance);
    }

    void PartitionerTest::test_find_split_depth2() {
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
        const size_t nfeatures = 2;
        const size_t depth = 2;

        SequentialSampler sampler(0, ys.size());
        Partitioner builder(nfeatures, depth, rmse, xs, ys);
        builder.build(sampler);

        CPPUNIT_ASSERT(0 == builder.m_feature_idxs[1]);
        CPPUNIT_ASSERT(0 == builder.m_feature_idxs[2]);
        CPPUNIT_ASSERT(1 == builder.m_feature_idxs[3]);

        // via hand-calculation
        // top-level split on feature 0 at value 5.65
        // LHS second level split on feature 0 at value 5.20
        // RHS second level split on feature 1 at value 6.01
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.65f, builder.m_split_vals[1], _tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.20f, builder.m_split_vals[2], _tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(6.01f, builder.m_split_vals[3], _tolerance);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(19.335f, builder.m_predictions.find(4)->second, _tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(14.59f, builder.m_predictions.find(5)->second, _tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(20.75f, builder.m_predictions.find(6)->second, _tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(23.49f, builder.m_predictions.find(7)->second, _tolerance);
    }

    void PartitionerTest::test_rmse() {
        std::vector<float> left { 8.0f, 2.5f, 4.5f };
        std::vector<float> right { 18.4f, 0.0f, -12.4f, -36.2f };
        double err = rmse(left, right);
        double expected = 39.878;

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, err, 1e-3);
    }

}

int main(int argc, char **argv) {
    CppUnit::TextUi::TestRunner runner;
    CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
    runner.addTest( registry.makeTest() );
    runner.run();
    return 0;
}
