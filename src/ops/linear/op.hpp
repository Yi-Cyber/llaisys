#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
    // Linear 算子: out = input @ weight.T + bias
    // bias 可以是 nullptr (如果是 tensor_t 智能指针，则是空指针)
    void linear(tensor_t out, tensor_t input, tensor_t weight, tensor_t bias);
}