#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
    // Rearrange: 将 in 张量的数据复制到 out 张量
    // 要求 out 和 in 形状相同，但步长 (strides) 可以不同
    void rearrange(tensor_t out, tensor_t in);
}