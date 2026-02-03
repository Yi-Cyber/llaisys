#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {
    void embedding_cuda(tensor_t out, tensor_t index, tensor_t weight);
}