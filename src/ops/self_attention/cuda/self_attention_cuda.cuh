#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {
    void self_attention_cuda(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
}