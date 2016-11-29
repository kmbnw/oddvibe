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

#include <vector>
#include <cppunit/extensions/HelperMacros.h>

#ifndef KMBNW_ODVB_MATHX_TEST_H
#define KMBNW_ODVB_MATHX_TEST_H

namespace oddvibe {
    class MathXTest : public CppUnit::TestFixture {
        CPPUNIT_TEST_SUITE(MathXTest);
        CPPUNIT_TEST(test_normalize);
        CPPUNIT_TEST(test_normalize_gt_one);
        CPPUNIT_TEST(test_normalize_lt_one);
        CPPUNIT_TEST_SUITE_END();

        private:
            const float m_tolerance = 1e-6;

        public:
            void setUp();
            void tearDown();
            void test_fit();
            void test_normalize();
            void test_normalize_gt_one();
            void test_normalize_lt_one();
    };
}
#endif
