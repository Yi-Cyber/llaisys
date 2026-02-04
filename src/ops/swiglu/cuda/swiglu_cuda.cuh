#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {
    void swiglu_cuda(tensor_t out, tensor_t gate, tensor_t up);
}