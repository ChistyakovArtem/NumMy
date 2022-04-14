//
// Created by artem on 11.04.2022.
//

#ifndef H_STYLE_TESTS_H
#define H_STYLE_TESTS_H

#include "Tensor.h"

namespace Tests {
    // ------------------------------------------------------------------------------------------- passed tests
    void TestTranspose() {
        auto t = Tensor(VectorOperations::arange(0, 24, 1), {2, 3, 4});
        t = t.transpose({2, 0, 1});
        t.make_flatten();
        t.print();
    }
    void TestPlusOperators() {
        auto t1 = Tensor(VectorOperations::arange(0, 24, 1), {2, 3, 4});
        auto t2 = Tensor(VectorOperations::arange(0, 48, 2), {2, 3, 4});

        t1.print("t1 = ");
        t2.print("t2 = ");

        auto t3 = t1 + t2;
        t3.print("t3 = ");

        t3 = t1 + 5;
        t3.print("t1 + 5 = ");


        Tensor t4 = t1;
        t4 += t2;
        t4.print("t1 += t2");

        Tensor t5 = t1;
        t5 += 5;
        t5.print("t1 += 5");
    }

    void TestMatMul() {
        auto t1 = Tensor(VectorOperations::arange(0, 24, 1), {4, 6});
        auto t2 = Tensor(VectorOperations::arange(0, 18, 1), {6, 3});
        auto t3 = t1 ^ t2;
        std::cout << t3 << std::endl;
    }

    void TestReshape() {
        auto t1 = Tensor(0, 24, 1);
        t1.make_reshaped({4, 6});
        t1.print();
    }
    void TestStaticCast() {
        auto t1 = Tensor(0, 1, 1);
        std::cout << static_cast<int>(t1) << std::endl;
    }
    void TestMultiplication() {
        auto t1 = Tensor(0, 24, 1);
        t1.make_reshaped({4, 6});
        t1 *= 2;
        t1.print();
    }
    void TestSumMeanMax() {
        auto t = Tensor(0, 120, 1);
        t.make_reshaped({2, 3, 4, 5});

        t.sum({0, 2}).print();
        t.astype<double>().mean({0, 2}).print();
        t.max({0, 2}).print();
    }
    void RandomUniform() {
        auto t = Tensor<int>::random_uniform({1000}, -1.0, 1.0, {11});
        (t == 0).astype<double>().mean({}).print();
        (t == 1).astype<double>().mean({}).print();
        (t == -1).astype<double>().mean({}).print();
    }
    void Max2() {
        auto t = Tensor(-10, 10, 1);
        auto t1 = t;
        t1.reverse();
        t.max(t1);
        t.print();
    }
    void TestSlices() {
        auto t = Tensor(0, 24, 1);
        t.make_reshaped({4, 6});
        t.print();
        t = t.get_slices({{1}, {1, 5}});
        t.print();
    }
    // ------------------------------------------------------------------------------------------- non-passed tests

}

#endif //H_STYLE_TESTS_H
