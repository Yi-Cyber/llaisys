#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {
    void linear_cuda(tensor_t out, tensor_t input, tensor_t weight, tensor_t bias);
}