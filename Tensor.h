//
// Created by artem on 11.04.2022.
//

#ifndef H_STYLE_TENSOR_H
#define H_STYLE_TENSOR_H

#include <vector>
#include <iostream>
#include <random>
#include <typeinfo>
#include <queue>

#include "VectorOperations.h"

template<class ValuesType>
class Tensor {

public:

    // ------------------------------------------------------------------------------------------------- constructors

    Tensor(const std::vector<ValuesType> &values, const std::vector<size_t> &axis);
    Tensor(ValuesType start, ValuesType finish, ValuesType step);

    // ------------------------------------------------------------------------------------------------- classic functions

    Tensor<ValuesType> transpose(const std::vector<size_t> &axis_permutation);
    Tensor<ValuesType> sum(const std::vector<size_t> &axis_indices_to_sum);
    Tensor<ValuesType> mean(const std::vector<size_t> &axis_indices_to_mean);
    Tensor<ValuesType> max(const std::vector<size_t> &axis_indices_to_max);

    Tensor<ValuesType> get_slices(const std::vector<std::vector<size_t> >& slices);
    template<class T>
    Tensor<T> astype();

    // ------------------------------------------------------------------------------------------------- make-functions

    void make_flatten();
    void make_reshaped(const std::vector<size_t>& new_axis);

    void max(ValuesType b);
    void max(const Tensor<ValuesType> &b);

    void reverse();

    // ------------------------------------------------------------------------------------------------- random

    template<typename T>
    static Tensor<T> random_uniform(const std::vector<size_t>& axis, T start, T finish, std::seed_seq seedSeq);
    static Tensor<uint8_t> random_binomial(const std::vector<size_t>& axis, double threshold, std::seed_seq seedSeq);

    // ------------------------------------------------------------------------------------------------- operators

    template<typename T>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& t);
    void print(const std::string& prefix="");

    [[nodiscard]] ValuesType at(const std::vector<size_t>& a) const;
    explicit operator ValuesType() const { return values_[0]; }

    Tensor<ValuesType> operator+(const Tensor<ValuesType>& b);
    Tensor<ValuesType> operator+(ValuesType b);
    Tensor<ValuesType> operator+=(const Tensor<ValuesType>& b);
    Tensor<ValuesType> operator+=(ValuesType b);

    Tensor<ValuesType> operator-(const Tensor<ValuesType>& b);
    Tensor<ValuesType> operator-(ValuesType b);
    Tensor<ValuesType> operator-=(const Tensor<ValuesType>& b);
    Tensor<ValuesType> operator-=(ValuesType b);
    Tensor<ValuesType> operator-();

    Tensor<ValuesType> operator*(const Tensor<ValuesType>& b);
    Tensor<ValuesType> operator*(ValuesType b);
    Tensor<ValuesType> operator*=(const Tensor<ValuesType>& b);
    Tensor<ValuesType> operator*=(ValuesType b);

    Tensor<ValuesType> operator/(const Tensor<ValuesType>& b);
    Tensor<ValuesType> operator/(ValuesType b);
    Tensor<ValuesType> operator/=(const Tensor<ValuesType>& b);
    Tensor<ValuesType> operator/=(ValuesType b);

    Tensor<ValuesType> operator^(const Tensor<ValuesType>& b);

    Tensor<uint8_t> operator==(ValuesType b);
    Tensor<uint8_t> operator==(const Tensor<ValuesType>& b);
    bool equal(const Tensor<ValuesType>& b);

private:
    std::vector<ValuesType> values_;
    std::vector<size_t> axis_;
    std::vector<size_t> strides_;
};

// ------------------------------------------------------------------------------------------------------ constructors

template<class ValuesType>
Tensor<ValuesType>::Tensor(const std::vector<ValuesType> &values, const std::vector<size_t> &axis) {
    values_ = values;
    axis_ = axis;
    strides_ = VectorOperations::GetStridesByAxis(axis_);
}

template<class ValuesType>
Tensor<ValuesType>::Tensor(ValuesType start, ValuesType finish, ValuesType step) {
    values_ = VectorOperations::arange(start, finish, step);
    axis_ = {values_.size()};
    strides_ = VectorOperations::GetStridesByAxis(axis_);
}

// ------------------------------------------------------------------------------------------------------ classic operations

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::transpose(const std::vector<size_t>& axis_permutation) {
    std::vector<size_t> new_axis = VectorOperations::GetValuesByIndices(axis_, axis_permutation);
    Tensor<ValuesType> answer(values_, new_axis);

    for (size_t i = 0; i < values_.size(); ++i) {
        std::vector<size_t> decomposition = VectorOperations::DecomposeToAxis(i, strides_);

        std::vector<size_t> new_decomposition =
                VectorOperations::GetValuesByIndices(decomposition, axis_permutation);

        size_t new_index = VectorOperations::ScalarProduct(answer.strides_, new_decomposition);
        answer.values_[new_index] = values_[i];
    }
    return answer;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::sum(const std::vector<size_t> &axis_indices_to_sum) {
    if (axis_indices_to_sum.empty()) {
        ValuesType ans = 0;
        for (auto i : values_) {
            ans += i;
        }
        return Tensor({ans}, {1});
    }

    std::vector<size_t> new_axis = VectorOperations::GetValuesByIndices(axis_, axis_indices_to_sum);
    std::vector<size_t> new_strides = VectorOperations::GetStridesByAxis(new_axis);
    size_t new_values_size = VectorOperations::GetInnerProduct(new_axis);

    std::vector<ValuesType> new_values(new_values_size);
    size_t compress_number = values_.size() / new_values.size();

    std::vector<size_t> new_perm =
            VectorOperations::PermuteAxisForOperation(axis_indices_to_sum, axis_.size());

    Tensor t = transpose(new_perm);

    ValuesType tmp = 0;
    for (size_t i = 0; i < t.values_.size(); ++i) {
        new_values[i / compress_number] += t.values_[i];
    }
    return Tensor(new_values, new_axis);
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::mean(const std::vector<size_t> &axis_indices_to_mean) {
    if (axis_indices_to_mean.empty()) {
        ValuesType ans = 0;
        for (auto i : values_) {
            ans += i / values_.size();
        }
        return Tensor({ans}, {1});
    }

    std::vector<size_t> new_axis = VectorOperations::GetValuesByIndices(axis_, axis_indices_to_mean);
    std::vector<size_t> new_strides = VectorOperations::GetStridesByAxis(new_axis);
    size_t new_values_size = VectorOperations::GetInnerProduct(new_axis);

    std::vector<ValuesType> new_values(new_values_size);
    size_t compress_number = values_.size() / new_values.size();

    std::vector<size_t> new_perm =
            VectorOperations::PermuteAxisForOperation(axis_indices_to_mean, axis_.size());

    Tensor t = transpose(new_perm);

    ValuesType tmp = 0;
    for (size_t i = 0; i < t.values_.size(); ++i) {
        new_values[i / compress_number] += t.values_[i];
    }
    return Tensor(new_values, new_axis) / compress_number;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::max(const std::vector<size_t> &axis_indices_to_max) {
    /**
     * Calculates max along axises
     */
    if (axis_indices_to_max.empty()) {
        ValuesType ans = values_[0];
        for (auto i : values_) {
            ans = std::max(ans, i);
        }
        return Tensor({ans}, {1});
    }

    std::vector<size_t> new_axis = VectorOperations::GetValuesByIndices(axis_, axis_indices_to_max);
    std::vector<size_t> new_strides = VectorOperations::GetStridesByAxis(new_axis);
    size_t new_values_size = VectorOperations::GetInnerProduct(new_axis);

    std::vector<ValuesType> new_values(new_values_size);
    size_t compress_number = values_.size() / new_values.size();

    std::vector<size_t> new_perm =
            VectorOperations::PermuteAxisForOperation(axis_indices_to_max, axis_.size());

    Tensor t = transpose(new_perm);

    ValuesType tmp = 0;
    for (size_t i = 0; i < t.values_.size(); ++i) {
        ValuesType mx = t.values_[i];
        for (size_t j = 0; j < compress_number; ++j) {
            mx = std::max(mx, t.values_[i + j]);
        }
        new_values[i / compress_number] = mx;
        i += compress_number - 1;
    }
    return Tensor(new_values, new_axis);
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::get_slices(const std::vector<std::vector<size_t> >& slices) {
    std::vector<size_t> new_axis(axis_.size());
    for (size_t i = 0; i < slices.size(); ++i) {
        size_t begin;
        size_t end;
        if (slices[i].empty()) {
            // pick all
            begin = 0;
            end = axis_[i];
        } else if (slices[i].size() == 1) {
            // pick 1 el
            begin = slices[i][0];
            end = slices[i][0] + 1;
        } else if (slices[i].size() == 2) {
            // pick 2 el
            begin = slices[i][0];
            end = slices[i][1];
        }
        new_axis[i] = end - begin;
    }

    std::vector<ValuesType> new_values;
    new_values.reserve(VectorOperations::GetInnerProduct(new_axis));

    std::queue<std::vector<size_t> > queue;
    queue.push({});
    do {
        std::vector<size_t> curr_axes = queue.front();
        queue.pop();
        size_t curr_axis = curr_axes.size();

        if (curr_axis == slices.size()) { // ( slices.size() = axis.size())
            new_values.push_back(VectorOperations::ScalarProduct(curr_axes, strides_));
        } else {
            auto vct = slices[curr_axis];
            size_t begin;
            size_t end;
            if (vct.empty()) {
                // pick all
                begin = 0;
                end = axis_[curr_axis];
            } else if (vct.size() == 1) {
                // pick 1 el
                begin = vct[0];
                end = vct[0] + 1;
            } else if (vct.size() == 2) {
                // pick 2 el
                begin = vct[0];
                end = vct[1];
            }

            for (size_t i = begin; i < end; ++i) {
                curr_axes.push_back(i);
                queue.push(curr_axes);
                curr_axes.pop_back();
            }
        }
    } while (! queue.empty());

    return Tensor<ValuesType>(new_values, new_axis);
}

template<class ValuesType>
template<class T>
Tensor<T> Tensor<ValuesType>::astype() {
    std::vector<T> new_values;
    new_values.reserve(values_.size());
    for (auto i : values_) {
        new_values.push_back(static_cast<T>(i));
    }
    return Tensor<T>(new_values, axis_);
}

// ------------------------------------------------------------------------------------------------------ make-operations

template<class ValuesType>
void Tensor<ValuesType>::make_flatten() {
    axis_ = {values_.size()};
    strides_ = VectorOperations::GetStridesByAxis(axis_);
}

template<class ValuesType>
void Tensor<ValuesType>::make_reshaped(const std::vector<size_t> &new_axis) {
    axis_ = new_axis;
    strides_ = VectorOperations::GetStridesByAxis(axis_);
}

template<class ValuesType>
void Tensor<ValuesType>::max(ValuesType b) {
    for (size_t i = 0; i < this->values_.size(); ++i) {
        this->values_[i] = std::max(this->values_[i], b);
    }
}

template<class ValuesType>
void Tensor<ValuesType>::max(const Tensor<ValuesType> &b) {
    for (size_t i = 0; i < this->values_.size(); ++i) {
        this->values_[i] = std::max(this->values_[i], b.values_[i]);
    }
}

template<class ValuesType>
void Tensor<ValuesType>::reverse() {
    std::reverse(this->values_.begin(), this->values_.end());
}


// ------------------------------------------------------------------------------------------------------ OPERATORS

template<class ValuesType>
ValuesType Tensor<ValuesType>::at(const std::vector<size_t>& a) const {
    return values_[VectorOperations::ScalarProduct(a, strides_)];
}

// ------------------------------------------------------------------------------------------------------ output

template<typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &t) {
    for (size_t i = 0; i < t.values_.size(); ++i) {
        os << t.values_[i] << " ";
        for (size_t j = 0; j < t.strides_.size() - 1; ++j) {
            if ((i + 1) % (t.strides_[j]) == 0) {
                os << std::endl;
            }
        }
    }
    return os;
}

template<class ValuesType>
void Tensor<ValuesType>::print(const std::string& prefix) {
    if (! prefix.empty()) {
        std::cout << prefix << std::endl;
    }
    for (size_t i = 0; i < values_.size(); ++i) {
        std::cout << +values_[i] << " ";
        for (size_t j = 0; j < strides_.size() - 1; ++j) {
            if ((i + 1) % (strides_[j]) == 0) {
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

// ------------------------------------------------------------------------------------------------------ random

template<class ValuesType>
template<typename T>
Tensor<T> Tensor<ValuesType>::random_uniform(const std::vector<size_t> &axis, T start, T finish, std::seed_seq seedSeq) {

    std::mt19937 eng(seedSeq);
    if (typeid(start).name() == "d") {
        std::uniform_real_distribution<> dis(start, finish);
        std::vector<T> new_values(VectorOperations::GetInnerProduct(axis));
        for (T & i : new_values) {
            i = dis(eng);
        }
        return Tensor<T>(new_values, axis);
    } else {
        std::uniform_int_distribution<> dis(start, finish - 1);
        std::vector<T> new_values(VectorOperations::GetInnerProduct(axis));
        for (T & i : new_values) {
            i = dis(eng);
        }
        return Tensor<T>(new_values, axis);
    }
}

template<class ValuesType>
Tensor<uint8_t> Tensor<ValuesType>::random_binomial(const std::vector<size_t> &axis, double threshold, std::seed_seq seedSeq) {
    std::mt19937 gen(seedSeq);
    std::binomial_distribution<> dis(1, threshold);

    std::vector<uint8_t> new_values(VectorOperations::GetInnerProduct(axis));
    for (auto & i : new_values) {
        i = dis(gen);
        // i = rand();
    }

    return Tensor<uint8_t>(new_values, axis);
}

// ------------------------------------------------------------------------------------------------------ plus

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator+(const Tensor<ValuesType> &b) {
    Tensor<ValuesType> answer = *this;
    for (size_t i = 0; i < b.values_.size(); ++i) {
        answer.values_[i] += b.values_[i];
    }
    return answer;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator+(const ValuesType b) {
    Tensor<ValuesType> answer = *this;
    for (size_t i = 0; i < values_.size(); ++i) {
        answer.values_[i] += b;
    }
    return answer;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator+=(const Tensor<ValuesType> &b) {
    for (size_t i = 0; i < b.values_.size(); ++i) {
        this->values_[i] += b.values_[i];
    }
    return *this;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator+=(ValuesType b) {
    for (size_t i = 0; i < values_.size(); ++i) {
        this->values_[i] += b;
    }
    return *this;
}
// ------------------------------------------------------------------------------------------------------ minus

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator-(const Tensor<ValuesType> &b) {
    Tensor<ValuesType> answer = *this;
    for (size_t i = 0; i < b.values_.size(); ++i) {
        answer.values_[i] -= b.values_[i];
    }
    return answer;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator-(const ValuesType b) {
    Tensor<ValuesType> answer = *this;
    for (size_t i = 0; i < values_.size(); ++i) {
        answer.values_[i] -= b;
    }
    return answer;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator-=(const Tensor<ValuesType> &b) {
    for (size_t i = 0; i < b.values_.size(); ++i) {
        this->values_[i] -= b.values_[i];
    }
    return *this;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator-=(ValuesType b) {
    for (size_t i = 0; i < values_.size(); ++i) {
        this->values_[i] -= b;
    }
    return *this;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator-() {
    std::vector<ValuesType> new_values;
    new_values.reserve(values_.size());
    for (auto i : values_) {
        new_values.push_back(-i);
    }
}

// ------------------------------------------------------------------------------------------------------ multiplication

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator*(const Tensor<ValuesType> &b) {
    Tensor<ValuesType> answer = *this;
    for (size_t i = 0; i < b.values_.size(); ++i) {
        answer.values_[i] *= b.values_[i];
    }
    return answer;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator*(const ValuesType b) {
    Tensor<ValuesType> answer = *this;
    for (size_t i = 0; i < values_.size(); ++i) {
        answer.values_[i] *= b;
    }
    return answer;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator*=(const Tensor<ValuesType> &b) {
    for (size_t i = 0; i < b.values_.size(); ++i) {
        this->values_[i] *= b.values_[i];
    }
    return *this;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator*=(ValuesType b) {
    for (size_t i = 0; i < values_.size(); ++i) {
        this->values_[i] *= b;
    }
    return *this;
}

// ------------------------------------------------------------------------------------------------------ division

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator/(const Tensor<ValuesType> &b) {
    Tensor<ValuesType> answer = *this;
    for (size_t i = 0; i < b.values_.size(); ++i) {
        answer.values_[i] /= b.values_[i];
    }
    return answer;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator/(const ValuesType b) {
    Tensor<ValuesType> answer = *this;
    for (size_t i = 0; i < values_.size(); ++i) {
        answer.values_[i] /= b;
    }
    return answer;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator/=(const Tensor<ValuesType> &b) {
    for (size_t i = 0; i < b.values_.size(); ++i) {
        this->values_[i] /= b.values_[i];
    }
    return *this;
}

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator/=(ValuesType b) {
    for (size_t i = 0; i < values_.size(); ++i) {
        this->values_[i] /= b;
    }
    return *this;
}

// ------------------------------------------------------------------------------------------------------ equal

template<class ValuesType>
Tensor<uint8_t> Tensor<ValuesType>::operator==(ValuesType b) {
    std::vector<uint8_t> new_values;
    new_values.reserve(values_.size());
    for (auto i : values_) {
        if (i == b) {
            new_values.push_back(1);
        } else {
            new_values.push_back(0);
        }
    }
    return Tensor<uint8_t>(new_values, axis_);
}

template<class ValuesType>
Tensor<uint8_t> Tensor<ValuesType>::operator==(const Tensor<ValuesType> &b) {
    std::vector<uint8_t> new_values;
    new_values.reserve(values_.size());
    for (size_t i = 0; i < values_.size(); ++i) {
        if (values_[i] == b.values_[i]) {
            new_values.push_back(1);
        } else {
            new_values.push_back(0);
        }
    }
    return Tensor<uint8_t>(new_values, axis_);
}

template<class ValuesType>
bool Tensor<ValuesType>::equal(const Tensor<ValuesType> &b) {
    for (size_t i = 0; i < values_.size(); ++i) {
        if (values_[i] != b.values_[i]) {
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------------ matmul

template<class ValuesType>
Tensor<ValuesType> Tensor<ValuesType>::operator^(const Tensor<ValuesType> &b) {
    std::vector<size_t> new_axis = {this->axis_[0], b.axis_[1]};
    std::vector<ValuesType> new_values(VectorOperations::GetInnerProduct(new_axis));

    auto answer = Tensor(new_values, new_axis);
    auto tmp = *this;

    for (size_t row = 0; row < new_axis[0]; ++row) {
        for (size_t col = 0; col < new_axis[1]; ++col) {
            ValuesType sum = 0;
            for (size_t i = 0; i < b.axis_[0]; ++i) {
                sum += tmp.at({row, i}) * b.at({i, col});
            }
            answer.values_[row * new_axis[1] + col] = sum;
        }
    }
    return answer;
}


#endif //H_STYLE_TENSOR_H
