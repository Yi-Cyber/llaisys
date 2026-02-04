#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
    void rms_norm_cpu(tensor_t out, tensor_t input, tensor_t weight, float eps);
}