#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {
    void rms_norm_cuda(tensor_t out, tensor_t input, tensor_t weight, float eps);
}