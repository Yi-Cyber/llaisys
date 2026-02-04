#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
    // SwiGLU: out = up * (gate / (1 + exp(-gate)))
    void swiglu(tensor_t out, tensor_t gate, tensor_t up);
}