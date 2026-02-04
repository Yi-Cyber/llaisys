#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
    void rearrange_cpu(tensor_t out, tensor_t in);
}