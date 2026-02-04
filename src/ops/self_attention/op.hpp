#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
    // Self Attention: Output = Softmax(Q * K^T * scale) * V
    // 支持 GQA (Grouped Query Attention)
    void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
}