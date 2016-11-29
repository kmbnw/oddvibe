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
#include "partitioner.h"
#include "seq_sampler.h"
#include "cached_sampler.h"
#include "ecdf_sampler.h"
#include "regression_tree.h"
#include "booster.h"
#include "booster_test.h"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION(oddvibe::BoosterTest);

namespace oddvibe {

    void BoosterTest::setUp() {
    }

    void BoosterTest::tearDown() {
    }

    void BoosterTest::test_add_counts_cached() {
        const std::vector<float> pmf { 0.4f, 0.25f, 0.15f, 0.20f };
        EmpiricalSampler sampler(time(0), pmf);

        const size_t nrows = pmf.size();
        const size_t max_iter = 1000;
        std::vector<unsigned int> expected_counts(nrows, 0);

        CachedSampler cache(sampler);

        for (size_t i = 0; i < max_iter; ++i) {
            for(size_t k = 0; k < nrows; ++k) {
                expected_counts[cache.next_sample()]++;
            }
        }

        std::vector<unsigned int> counts(nrows, 0);
        for (size_t i = 0; i < max_iter; ++i) {
            add_counts(cache, counts);
        }

        for (size_t k = 0; k < nrows; ++k) {
            CPPUNIT_ASSERT(expected_counts[k] == counts[k]);
        }

        // expect sequential calls to give us the same values from the CachedSampler
        std::vector<unsigned int> counts2(nrows, 0);
        for (size_t i = 0; i < max_iter; ++i) {
            add_counts(cache, counts2);
        }

        for (size_t k = 0; k < nrows; ++k) {
            CPPUNIT_ASSERT(expected_counts[k] == counts2[k]);
        }
    }

    void BoosterTest::test_add_counts() {
        const std::vector<float> pmf { 0.4f, 0.25f, 0.15f, 0.20f };
        EmpiricalSampler sampler(time(0), pmf);

        const size_t nrows = pmf.size();
        const size_t max_iter = 100000;
        const size_t sample_sz = max_iter * nrows;

        std::vector<unsigned int> counts(nrows, 0);
        for (size_t i = 0; i < max_iter; ++i) {
            add_counts(sampler, counts);
        }

        // should have about the same distribution as the initial PMF
        for (size_t k = 0; k != counts.size(); ++k) {
            // note: this could technically fail, but it is quite unlikely
            CPPUNIT_ASSERT(counts[k] > 0);

            float pct = (float) (((double) counts[k]) / sample_sz);
            //std::cout << "X == " << k << ": " << pct << std::endl;
            CPPUNIT_ASSERT_DOUBLES_EQUAL(pmf[k], pct, 1e-2);
        }
    }

    void BoosterTest::test_fit() {
        // very simple linear regression, y = B0 + B1 * x
        // where Y = 2.4 + 1.5 * (X + noise)
        // fitting without an intercept using R's lm()
        // function gives a coefficent of 1.738
        // fitting with the intercept gives 2.457 + 1.5 * X

        const std::vector<float> xs {
            4.859846f,
            5.808777f,
            7.241938f,
            7.988773f,
            8.910130f,
            10.241526f,
            10.773271f,
            11.897482f,
            13.120511f,
            13.778213f
        };

        const std::vector<float> ys {
            9.9f,
            11.4f,
            12.9f,
            14.4f,
            15.9f,
            17.4f,
            18.9f,
            20.4f,
            21.9f,
            23.4f
        };
        const size_t nfeatures = 1;
        const size_t depth = 1;
        const size_t num_rounds = 500;

        Partitioner builder(nfeatures, depth, xs, ys);
        const Booster fitter((Partitioner* const) &builder, time(0), rmse);
        std::vector<float> pmf;
        std::vector<unsigned int> counts;

        for (size_t k = 0; k < num_rounds; ++k) {
            fitter.update_one(pmf, counts, xs, ys);
        }

/*
        std::vector<float> yhats;
        tree.predict(xs, yhats);

        // split is done on feature 0, split value 3.4
        const float left_leaf = 5.25f;
        const float right_leaf = -18.1f;

        CPPUNIT_ASSERT_DOUBLES_EQUAL(left_leaf, yhats[0], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(left_leaf, yhats[1], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(right_leaf, yhats[2], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(right_leaf, yhats[3], m_tolerance);*/
    }

    void BoosterTest::test_predict_depth2() {
/*        const std::vector<float> xs {
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

        SequentialSampler sampler(0, ys.size());
        Partitioner builder(nfeatures, depth, xs, ys);
        builder.build(sampler);

        const RegressionTree tree(builder);

        std::vector<float> yhats;
        tree.predict(new_xs, yhats);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[0], yhats[0], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[1], yhats[1], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[2], yhats[2], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[3], yhats[3], m_tolerance);*/
    }
}
