#pragma once

#include "../../../tensor/tensor.hpp"
#include <cuda_runtime.h>

namespace llaisys::ops::cuda {

/**
 * @brief CUDA Argmax 入口
 * @param stream CUDA 流 (用于异步执行)
 */
void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, 
            llaisysDataType_t dtype, size_t numel, cudaStream_t stream = 0);

} // namespace llaisys::ops::cuda