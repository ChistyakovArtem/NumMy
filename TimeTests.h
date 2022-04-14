//
// Created by artem on 14.04.2022.
//

#ifndef H_STYLE_TIMETESTS_H
#define H_STYLE_TIMETESTS_H
#include <chrono>
#include "Tensor.h"

namespace TimeTests {

    // ------------------------------------------------------------------------------------------- passed tests
    void Initialization() {
        auto start_time = std::chrono::steady_clock::now();

        auto a = Tensor(0, 100000000, 1);
        a.make_reshaped({10, 10, 10, 10, 10, 10, 10, 10});

        auto init_time = std::chrono::steady_clock::now();
        // code

        a = a.get_slices({{}, {}, {}, {}});
        // code

        auto end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> full_time = end_time - start_time;
        std::chrono::duration<double> work_time = end_time - init_time;

        std::cout << "Full time = " << (full_time).count() << std::endl;
        std::cout << "Work time = " << (work_time).count() << std::endl;
        a.sum({}).print();
    }

    void make_reshaped() {
        auto start_time = std::chrono::steady_clock::now();

        auto a = Tensor(0, 100000000, 1);
        a.make_reshaped({100, 100, 100, 100});

        auto init_time = std::chrono::steady_clock::now();
        // code
        for (int i = 0; i < 1000000; ++i) {
            a.make_reshaped({100, 100, 100, 100});
        }
        // code

        auto end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> full_time = end_time - start_time;
        std::chrono::duration<double> work_time = end_time - init_time;

        std::cout << "Full time = " << (full_time).count() << std::endl;
        std::cout << "Work time = " << (work_time).count() << std::endl;
        a.sum({}).print();
    }

    void random_uniform() {
        auto start_time = std::chrono::steady_clock::now();

        auto init_time = std::chrono::steady_clock::now();
        // code

        auto a = Tensor<int>::random_uniform({100, 100, 100, 100}, 0, 10, {});
        // code

        auto end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> full_time = end_time - start_time;
        std::chrono::duration<double> work_time = end_time - init_time;

        std::cout << "Full time = " << (full_time).count() << std::endl;
        std::cout << "Work time = " << (work_time).count() << std::endl;
        a.sum({}).print();
    }

    void random_binomial() {
        auto start_time = std::chrono::steady_clock::now();

        auto init_time = std::chrono::steady_clock::now();
        // code

        auto a = Tensor<int>::random_binomial({100, 100, 100, 100}, 0.5, {});
        // code

        auto end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> full_time = end_time - start_time;
        std::chrono::duration<double> work_time = end_time - init_time;

        std::cout << "Full time = " << (full_time).count() << std::endl;
        std::cout << "Work time = " << (work_time).count() << std::endl;
        a.astype<long long>().sum({}).print();
    }
    // ------------------------------------------------------------------------------------------- non-passed tests
}
#endif //H_STYLE_TIMETESTS_H
