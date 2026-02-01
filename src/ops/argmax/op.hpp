#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
/**
 * @brief 查找最大值及其索引
 * @param max_idx 输出张量：存储最大值的索引 (类型通常为 Int64)
 * @param max_val 输出张量：存储最大值本身 (类型与 vals 相同)
 * @param vals 输入张量：被搜索的数据 (1D)
 */
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
} // namespace llaisys::ops