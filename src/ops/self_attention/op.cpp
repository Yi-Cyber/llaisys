#include "op.hpp"

// 引入核心头文件
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#include "cuda/self_attention_cuda.cuh"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 1. 设备一致性
    CHECK_SAME_DEVICE(attn_val, q);
    CHECK_SAME_DEVICE(q, k);
    CHECK_SAME_DEVICE(k, v);

    // 2. 连续性
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), 
           "Self Attention: all tensors must be contiguous.");

    // 3. 维度解析与检查
    // q: [seqlen, nhead, d]
    // k: [total_len, nkvhead, d]
    // v: [total_len, nkvhead, dv]
    // attn_val: [seqlen, nhead, dv]
    
    size_t nhead = q->shape()[1];
    size_t nkvhead = k->shape()[1];
    
    ASSERT(nhead % nkvhead == 0, "Self Attention: nhead must be divisible by nkvhead (GQA).");
    ASSERT(q->shape()[2] == k->shape()[2], "Self Attention: Q and K must have same head_dim.");
    ASSERT(v->shape()[2] == attn_val->shape()[2], "Self Attention: V and Output must have same head_dim.");

    // 4. CPU 分发
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention_cpu(attn_val, q, k, v, scale);
    }

    // 5. 设置上下文并分发
    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention_cpu(attn_val, q, k, v, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::self_attention_cuda(attn_val, q, k, v, scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops