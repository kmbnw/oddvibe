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
#include <cmath>
#include <vector>
#include <unordered_map>
#include "../../src/math_x.h"
#include "math_x_test.h"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION(oddvibe::MathXTest);

namespace oddvibe {

    void MathXTest::setUp() {
    }

    void MathXTest::tearDown() {
    }

    void MathXTest::test_normalize() {
        const std::vector<float> expected { 0.4, 0.4, 0.2 };
        std::vector<float> pmf { 0.4, 0.4, 0.2 };
        normalize(pmf);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[0], pmf[0], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[1], pmf[1], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[2], pmf[2], m_tolerance);
    }

    void MathXTest::test_normalize_gt_one() {
        const std::vector<float> expected { 0.2, 0.64, 0.16 };
        std::vector<float> pmf { 0.25, 0.8, 0.2 };
        normalize(pmf);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[0], pmf[0], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[1], pmf[1], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[2], pmf[2], m_tolerance);
    }

    void MathXTest::test_normalize_lt_one() {
        const std::vector<float> expected { 0.125, 0.5, 0.125, 0.25 };
        std::vector<float> pmf { 0.1, 0.4, 0.1, 0.2 };
        normalize(pmf);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[0], pmf[0], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[1], pmf[1], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[2], pmf[2], m_tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[3], pmf[3], m_tolerance);
    }
}

int main(int argc, char **argv) {
    CppUnit::TextUi::TestRunner runner;
    CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
    runner.addTest( registry.makeTest() );
    runner.run();
    return 0;
}
