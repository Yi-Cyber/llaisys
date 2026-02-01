#pragma once

#include "../../../tensor/tensor.hpp" // 需要用到 llaisysDataType_t
#include <cstddef> // size_t

namespace llaisys::ops::cpu {

/**
 * @brief CPU 端 Argmax 的具体实现入口
 * @param max_idx 输出内存指针 (存储索引)
 * @param max_val 输出内存指针 (存储数值)
 * @param vals 输入内存指针 (数据)
 * @param dtype 输入数据的数据类型 (Float32/Float16/BFloat16)
 * @param numel 输入数据的元素个数
 */
void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, llaisysDataType_t dtype, size_t numel);

} // namespace llaisys::ops::cpu