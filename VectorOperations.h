//
// Created by artem on 11.04.2022.
//

#ifndef H_STYLE_VECTOROPERATIONS_H
#define H_STYLE_VECTOROPERATIONS_H

#include <cstddef>
#include <vector>

namespace VectorOperations {
    std::vector<size_t> PermuteAxisForOperation(std::vector<size_t> axis_to_use, size_t axis_size);
    std::vector<size_t> GetStridesByAxis(const std::vector<size_t>& axis);
    std::vector<size_t> GetSuffixCumulativeProduct(const std::vector<size_t>& axis);
    std::vector<size_t> DecomposeToAxis(size_t index, const std::vector<size_t>& strides);
    std::vector<size_t> ReversePermutation(const std::vector<size_t>& permutation);

    template<typename T>
    T GetInnerProduct(const std::vector<T>& a);
    template<typename T>
    std::vector<T> GetValuesByIndices(const std::vector<T>& values, const std::vector<size_t>& indices);
    template<typename T>
    T ScalarProduct(const std::vector<T>& a, const std::vector<T>& b);

    // generators
    template<typename T>
    std::vector<T> arange(T start, T finish, T step);

};

template<typename T>
T VectorOperations::GetInnerProduct(const std::vector<T> &a) {
    T answer = 1;
    for (auto i : a) {
        answer *= i;
    }
    return answer;
}

std::vector<size_t> VectorOperations::PermuteAxisForOperation(std::vector<size_t> axis_to_use, size_t axis_size) {
    std::vector<bool> used(axis_size, false);
    for (auto i : axis_to_use) {
        used[i] = true;
    }

    for (size_t i = 0; i < axis_size; ++i) {
        if (! used[i]) {
            axis_to_use.push_back(i);
        }
    }

    return axis_to_use;
}

std::vector<size_t> VectorOperations::GetStridesByAxis(const std::vector<size_t> &axis) {
    std::vector<size_t> strides(axis.size());
    strides[axis.size() - 1] = 1;
    for (size_t i = axis.size() - 1; i >= 1; --i) {
        strides[i - 1] = strides[i] * axis[i];
    }
    return strides;
}

std::vector<size_t> VectorOperations::GetSuffixCumulativeProduct(const std::vector<size_t> &axis) {
    std::vector<size_t> answer(axis.size());
    answer[axis.size() - 1] = axis[axis.size() - 1];
    for (size_t i = 1; i < axis.size(); ++i) {
        answer[axis.size() - 1 - i] = answer[axis.size() - i] * axis[axis.size() - 1 - i];
    }
    return answer;
}

std::vector<size_t> VectorOperations::DecomposeToAxis(size_t index, const std::vector<size_t> &strides) {
    std::vector<size_t> decomposition(strides.size());
    for (size_t i = 0; i < strides.size(); ++i) {
        decomposition[i] = index / strides[i];
        index = index % strides[i];
    }
    return decomposition;
}

std::vector<size_t> VectorOperations::ReversePermutation(const std::vector<size_t> &permutation) {
    std::vector<size_t> answer(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        answer[permutation[i]] = i;
    }
    return answer;
}

template<typename T>
std::vector<T> VectorOperations::GetValuesByIndices(const std::vector<T> &values, const std::vector<size_t> &indices) {
    std::vector<T> answer;
    answer.reserve(values.size());
    for (size_t ind : indices) {
        answer.push_back(values[ind]);
    }
    return answer;
}

template<typename T>
T VectorOperations::ScalarProduct(const std::vector<T> &a, const std::vector<T> &b) {
    T answer = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        answer += a[i] * b[i];
    }
    return answer;
}

template<typename T>
std::vector<T> VectorOperations::arange(T start, T finish, T step) {
    std::vector<T> answer;
    answer.reserve((finish - start) / step);
    for (T el = start; el < finish; el += step) {
        answer.push_back(el);
    }
    return answer;
}

#endif //H_STYLE_VECTOROPERATIONS_H
