#include "argmax_cuda.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <limits>

// 定义 block 大小
constexpr int BLOCK_SIZE = 256;

// 辅助函数：将不同类型转换为 float 进行比较 (避免半精度比较的复杂性)
template <typename T>
__device__ float to_float(T val) {
    return static_cast<float>(val);
}

// 针对 fp16 的特化
template <>
__device__ float to_float<__half>(__half val) {
    return __half2float(val);
}

// 针对 bf16 的特化
template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

// --------------------------------------------------------
// 核心 Kernel: 寻找最大值和索引
// --------------------------------------------------------
template <typename T>
__global__ void argmax_kernel(int64_t* out_idx, T* out_val, const T* vals, size_t n) {
    // 共享内存：存储每个线程找到的局部最大值和索引
    __shared__ float s_max_vals[BLOCK_SIZE];
    __shared__ int64_t s_max_idxs[BLOCK_SIZE];

    int tid = threadIdx.x;
    
    // 1. 线程局部初始化
    // 初始化为极小值
    float local_max_val = -1e30f; // 或者使用 -INFINITY
    int64_t local_max_idx = -1;

    // 2. 网格跨步循环 (Grid-Stride Loop)
    // 这样设计使得即便数组长度 n > BLOCK_SIZE，也能处理完所有数据
    for (size_t i = tid; i < n; i += blockDim.x) {
        float val_f = to_float(vals[i]);
        // 如果值更大，或者值相等但索引更小 (保持稳定性)，则更新
        if (val_f > local_max_val || (val_f == local_max_val && local_max_idx == -1)) {
            local_max_val = val_f;
            local_max_idx = i;
        }
    }

    // 将局部结果写入共享内存
    s_max_vals[tid] = local_max_val;
    s_max_idxs[tid] = local_max_idx;

    __syncthreads();

    // 3. 共享内存归约 (Tree Reduction)
    // 复杂度 O(log BLOCK_SIZE)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float val1 = s_max_vals[tid];
            float val2 = s_max_vals[tid + stride];
            
            // 比较当前位置和跨步位置的值
            if (val2 > val1) {
                s_max_vals[tid] = val2;
                s_max_idxs[tid] = s_max_idxs[tid + stride];
            } else if (val2 == val1) {
                // 如果值相等，保留较小的索引
                if (s_max_idxs[tid + stride] < s_max_idxs[tid] && s_max_idxs[tid + stride] != -1) {
                    s_max_idxs[tid] = s_max_idxs[tid + stride];
                }
            }
        }
        __syncthreads();
    }

    // 4. 写回结果
    // 归约完成后，Thread 0 拥有全局最大值
    if (tid == 0) {
        *out_idx = s_max_idxs[0];
        // 我们需要把原始类型的最大值写回去，这里为了简单重新从全局内存读取一次，
        // 或者你可以选择在共享内存里存 T 类型而不是 float。
        // 为保证绝对正确，直接根据索引去全局内存拿原始值：
        if (s_max_idxs[0] != -1) {
            *out_val = vals[s_max_idxs[0]];
        } else {
            // 处理空数组等极端情况
             *out_idx = 0;
        }
    }
}

namespace llaisys::ops::cuda {

void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, 
            llaisysDataType_t dtype, size_t numel, cudaStream_t stream) {
    
    // 只需要启动 1 个 Block
    dim3 grid(1);
    dim3 block(BLOCK_SIZE);

    int64_t* idx_ptr = reinterpret_cast<int64_t*>(max_idx);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<float><<<grid, block, 0, stream>>>(
            idx_ptr,
            reinterpret_cast<float*>(max_val),
            reinterpret_cast<const float*>(vals),
            numel
        );
        break;
    
    case LLAISYS_DTYPE_F16:
        argmax_kernel<__half><<<grid, block, 0, stream>>>(
            idx_ptr,
            reinterpret_cast<__half*>(max_val),
            reinterpret_cast<const __half*>(vals),
            numel
        );
        break;

    case LLAISYS_DTYPE_BF16:
        argmax_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            idx_ptr,
            reinterpret_cast<__nv_bfloat16*>(max_val),
            reinterpret_cast<const __nv_bfloat16*>(vals),
            numel
        );
        break;

    default:
        // 在 GPU 上抛出异常比较麻烦，通常打印错误或不做处理
        // 实际工程中这里应该 check error
        break;
    }
}

} // namespace llaisys::ops::cuda