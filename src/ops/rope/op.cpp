#include "op.hpp"

// 【经验】引入核心头文件
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#include "cuda/rope_cuda.cuh"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 1. 设备一致性
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DEVICE(in, pos_ids);

    // 2. 连续性
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), 
           "RoPE: all tensors must be contiguous.");

    // 3. 维度检查
    // in: [seqlen, nhead, head_dim]
    // pos_ids: [seqlen]
    ASSERT(in->ndim() == 3, "RoPE: input must be 3D [seqlen, nhead, head_dim].");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D [seqlen].");
    ASSERT(in->shape()[0] == pos_ids->shape()[0], 
           "RoPE: input seqlen (dim 0) must match pos_ids size.");
    
    // RoPE 需要两两配对，所以 head_dim 必须是偶数
    size_t head_dim = in->shape()[2];
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even.");

    // 4. CPU 分发
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope_cpu(out, in, pos_ids, theta);
    }

    // 5. 设置上下文并分发
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope_cpu(out, in, pos_ids, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::rope_cuda(out, in, pos_ids, theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops