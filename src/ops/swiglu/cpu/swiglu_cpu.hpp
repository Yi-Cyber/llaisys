#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
    void swiglu_cpu(tensor_t out, tensor_t gate, tensor_t up);
}