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
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <functional>
#include "train_data.h"
#include "ecdf_sampler.h"
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

    void BoosterTest::test_fit() {
        // mixture distribution generated by two separate linear equations
        // but with the same noise

        const size_t seed = 1480561820L; //1480455481L; //time(0);
        const size_t nrows = 50;
        const size_t nfeatures = 2;
        const float intercept = 0.75f;
        const float beta_1 = 2.0f;
        const float beta_2 = 5.8f;

        std::cout << "Seed: " << seed << std::endl;
        std::normal_distribution<float> noise_dist(0.0, 1.0);
        std::mt19937 rand_engine(seed);

        auto gen = std::bind(noise_dist, rand_engine);
        std::vector<float> xs_noise(nrows * 2);
        std::generate(std::begin(xs_noise), std::end(xs_noise), gen);

        // nominally small x values are associated with small Y values
        // but we want to have this find the outliers
        // so mix it up
        std::normal_distribution<float> feature_x1_dist(5.0, 1.0);
        std::normal_distribution<float> feature_x2_dist(4000.3, 90.0);

        std::vector<float> xs(nrows * nfeatures, 0);
        std::vector<float> ys(nrows, 0);

        size_t threshold = (size_t) (0.7 * nrows);

        for (size_t k = 0; k != xs.size(); ++k) {
            if (k < (threshold * 2)) {
                xs[k] = feature_x1_dist(rand_engine);
            } else {
                xs[k] = feature_x2_dist(rand_engine);
            }
        }

        for (size_t k = 0, row_idx = 0; k != ys.size(); ++k, row_idx += nfeatures) {
            ys[k] = intercept + beta_1 * xs[row_idx] + beta_2 * xs[row_idx + 1];
            if (k < threshold) {
                if (k % 5 == 0) {
                    ys[k] = ys[k] * 1000 * row_idx;
                }
            }
        }

        std::transform(xs.begin(), xs.end(), xs_noise.begin(), xs.begin(), std::plus<float>());

        const size_t nrounds = 5000;

        const DataSet train_data(nfeatures, xs, ys);

        const Booster booster(seed);
        std::vector<float> pmf;
        std::vector<size_t> counts;
        std::vector<float> avg_counts(ys.size(), 0);

        const auto outlier = booster.fit(train_data, nrounds);
        const size_t expected_row = 27;

        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.63f, outlier.second, 1e-3);
        CPPUNIT_ASSERT_EQUAL(expected_row, outlier.first);
    }
}
