#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
    // RMS Normalization: out = (input * weight) / sqrt(mean(input^2) + eps)
    void rms_norm(tensor_t out, tensor_t input, tensor_t weight, float eps);
}