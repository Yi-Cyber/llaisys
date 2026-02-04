#include "op.hpp"

// 【经验】引入核心头文件
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#include "cuda/rms_norm_cuda.cuh"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t input, tensor_t weight, float eps) {
    // 1. 设备一致性检查
    CHECK_SAME_DEVICE(out, input);
    CHECK_SAME_DEVICE(input, weight);

    // 2. 连续性检查 (题目假设输入都是连续的)
    ASSERT(out->isContiguous() && input->isContiguous() && weight->isContiguous(), 
           "RMS Norm: all tensors must be contiguous.");

    // 3. 维度简单检查 (可选，为了稳健性)
    // input: [M, N], weight: [N]
    ASSERT(weight->ndim() == 1, "RMS Norm: weight must be 1D.");
    ASSERT(input->shape().back() == weight->shape()[0], 
           "RMS Norm: last dimension of input must match weight size.");

    // 4. CPU 分发
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm_cpu(out, input, weight, eps);
    }

    // 5. 设置上下文并分发
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm_cpu(out, input, weight, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::rms_norm_cuda(out, input, weight, eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops