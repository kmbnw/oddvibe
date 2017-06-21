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
#include <vector>
#include "ecdf_sampler.h"
#include "sampling_dist.h"
#include "ecdf_sampler_test.h"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION(oddvibe::EmpiricalSamplerTest);

namespace oddvibe {

    void EmpiricalSamplerTest::setUp() {
    }

    void EmpiricalSamplerTest::tearDown() {
    }

    void EmpiricalSamplerTest::test_next_sample() {
        const std::vector<float> probs { 0.4f, 0.25f, 0.15f, 0.20f };
        const SamplingDist pmf(probs);
        EmpiricalSampler sampler(time(0), pmf);
        std::array<size_t, 4> indexes;
        std::fill(indexes.begin(), indexes.end(), 0);

        const size_t sample_sz = 100000;
        for (size_t k = 0; k < sample_sz; ++k) {
            size_t idx = sampler.next_sample();
            indexes[idx] = indexes[idx] + 1;
        }

        for (size_t k = 0; k != indexes.size(); ++k) {
            float pct = (float) (((double) indexes[k]) / sample_sz);
            //std::cout << "X == " << k << ": " << pct << std::endl;
            CPPUNIT_ASSERT_DOUBLES_EQUAL(probs[k], pct, 1e-2);
        }
    }

    void EmpiricalSamplerTest::test_next_sample_zero_prob() {
        const std::vector<float> probs { 0.35f, 0.0f, 0.4f, 0.25f };
        const SamplingDist pmf(probs);
        EmpiricalSampler sampler(time(0), pmf);
        std::array<size_t, 4> indexes;
        std::fill(indexes.begin(), indexes.end(), 0);

        const size_t sample_sz = 100000;
        for (size_t k = 0; k < sample_sz; ++k) {
            size_t idx = sampler.next_sample();
            indexes[idx] = indexes[idx] + 1;
        }

        // should have no values here for a zero probability event
        CPPUNIT_ASSERT(indexes[1] == 0);

        for (size_t k = 0; k != indexes.size(); ++k) {
            float pct = (float) (((double) indexes[k]) / sample_sz);
            //std::cout << "X == " << k << ": " << pct << std::endl;
            CPPUNIT_ASSERT_DOUBLES_EQUAL(probs[k], pct, 1e-2);
        }
    }
}
