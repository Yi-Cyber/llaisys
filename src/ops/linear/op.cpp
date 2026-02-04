#include "op.hpp"

// 引入核心头文件
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#include "cuda/linear_cuda.cuh"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t input, tensor_t weight, tensor_t bias) {
    // 1. 检查必要输入的设备一致性
    CHECK_SAME_DEVICE(out, input);
    CHECK_SAME_DEVICE(input, weight);
    
    // 如果 bias 存在，也要检查
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    // 2. 检查内存连续性
    ASSERT(out->isContiguous() && input->isContiguous() && weight->isContiguous(), 
           "Linear: all tensors must be contiguous.");
    
    // 3. 优先处理 CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear_cpu(out, input, weight, bias);
    }

    // 4. 设置上下文并分发
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear_cpu(out, input, weight, bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::linear_cuda(out, input, weight, bias);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops