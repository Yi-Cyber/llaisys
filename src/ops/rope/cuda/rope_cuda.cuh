#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {
    void rope_cuda(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
}