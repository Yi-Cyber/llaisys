#include "op.hpp"

// 引入核心头文件
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"
#include "cuda/rearrange_cuda.cuh"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    // 1. 设备一致性
    CHECK_SAME_DEVICE(out, in);

    // 2. 形状与类型必须一致
    // Rearrange 不改变形状，只改变内存布局
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // 【注意】这里绝对不能检查 isContiguous()，因为本算子就是为了处理不连续的情况

    // 3. CPU 分发
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange_cpu(out, in);
    }

    // 4. 设置上下文并分发
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange_cpu(out, in);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::rearrange_cuda(out, in);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops