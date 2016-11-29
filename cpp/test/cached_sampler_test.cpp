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
#include <vector>
#include "ecdf_sampler.h"
#include "cached_sampler.h"
#include "cached_sampler_test.h"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION(oddvibe::CachedSamplerTest);

namespace oddvibe {

    void CachedSamplerTest::setUp() {
    }

    void CachedSamplerTest::tearDown() {
    }

    void CachedSamplerTest::test_caching() {
        const std::vector<float> pmf { 0.4f, 0.25f, 0.15f, 0.20f };
        EmpiricalSampler sampler(time(0), pmf);

        const size_t nrows = pmf.size();
        const size_t max_iter = 100;

        std::vector<float> sample(nrows, 0);
        CachedSampler cache(sampler);

        for(size_t k = 0; k != sample.size(); ++k) {
            sample[k] = cache.next_sample();
        }

        for (size_t i = 0; i < max_iter; ++i) {
            for(size_t k = 0; k != sample.size(); ++k) {
                // should be exactly equal as they are the same floating point
                CPPUNIT_ASSERT(sample[k] == cache.next_sample());
            }
        }
    }

}
