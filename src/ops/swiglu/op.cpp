#include "op.hpp"

// 引入核心头文件
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"
#include "cuda/swiglu_cuda.cuh"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 1. 设备一致性
    CHECK_SAME_DEVICE(out, gate);
    CHECK_SAME_DEVICE(gate, up);

    // 2. 连续性
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), 
           "SwiGLU: all tensors must be contiguous.");

    // 3. 形状检查
    // out, gate, up 必须具有相同的形状
    ASSERT(out->shape() == gate->shape(), "SwiGLU: out and gate shape mismatch.");
    ASSERT(gate->shape() == up->shape(), "SwiGLU: gate and up shape mismatch.");

    // 4. CPU 分发
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu_cpu(out, gate, up);
    }

    // 5. 设置上下文并分发
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu_cpu(out, gate, up);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::swiglu_cuda(out, gate, up);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops