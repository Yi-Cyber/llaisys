#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"

// 【新增】 引入 CUDA 头文件
#ifdef ENABLE_NVIDIA_API
#include "cuda/argmax_cuda.cuh"
#endif

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // ... (之前的检查代码保持不变) ...
    ASSERT(vals->deviceType() == max_idx->deviceType(), "Device mismatch");

    // CPU 分发
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    // 设置设备上下文
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // 【修改】 调用 CUDA 实现
        // stream() 是从 context 获取当前的 CUDA 流
        return cuda::argmax(
            max_idx->data(), 
            max_val->data(), 
            vals->data(), 
            vals->dtype(), 
            vals->numel(),
            llaisys::core::context().stream() 
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops