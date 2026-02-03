#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
    // 定义 embedding 算子接口
    void embedding(tensor_t out, tensor_t index, tensor_t weight);
}