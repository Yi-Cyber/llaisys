#include "op.hpp"

// 【新增】必须包含这些核心头文件以获取设备枚举和宏
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#include "cuda/embedding_cuda.cuh"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 检查设备一致性 (假设 index 也在同一设备，或者至少 out 和 weight 在同一设备)
    CHECK_SAME_DEVICE(out, weight);
    // index 可能是 CPU tensor，但在某些框架下要求 index 也在 GPU。这里先只检查 out 和 weight。

    // 形状和类型检查
    // out: [num_indices, hidden_dim]
    // weight: [vocab_size, hidden_dim]
    // index: [num_indices]
    ASSERT(out->isContiguous() && weight->isContiguous() && index->isContiguous(), 
           "Embedding: all tensors must be contiguous.");
    
    // 如果是 CPU，直接执行
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding_cpu(out, index, weight);
    }

    // 设置当前设备上下文
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding_cpu(out, index, weight);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // 目前返回未实现，或者调用 cuda 实现
        // TO_BE_IMPLEMENTED(); 
        return cuda::embedding_cuda(out, index, weight);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops