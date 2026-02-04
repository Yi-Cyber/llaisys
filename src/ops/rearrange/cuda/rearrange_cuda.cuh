#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {
    void rearrange_cuda(tensor_t out, tensor_t in);
}