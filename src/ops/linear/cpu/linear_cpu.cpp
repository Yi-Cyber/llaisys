#include "linear_cpu.hpp"
#include <cstring>
#include <stdexcept>
#include "../../../tensor/tensor.hpp"
#include "../../../core/llaisys_core.hpp" 
#include "../../../utils.hpp" // 用于 utils::cast

namespace llaisys::ops::cpu {

// 辅助函数：将任意类型转换为 float 进行计算
template<typename T>
float to_float(T val) {
    // 如果 T 本身就是 float，直接返回；如果是 fp16/bf16，使用 utils::cast 或 static_cast
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        return utils::cast<float>(val);
    }
}

// 辅助函数：将 float 转回 T
template<typename T>
T from_float(float val) {
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        return utils::cast<T>(val);
    }
}

template <typename T>
void linear_kernel(tensor_t out, tensor_t input, tensor_t weight, tensor_t bias) {
    // 1. 获取数据指针
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* in_ptr = reinterpret_cast<const T*>(input->data());
    const T* w_ptr = reinterpret_cast<const T*>(weight->data());
    const T* b_ptr = bias ? reinterpret_cast<const T*>(bias->data()) : nullptr;

    // 2. 解析维度
    // Input: [M, K] (M = batch_size flattened, K = in_features)
    // Weight: [N, K] (N = out_features, K = in_features) -> 注意 PyTorch Linear 权重是转置存储的
    // Output: [M, N]
    
    // 我们假设输入的最后两维符合矩阵乘法规则，前面的维度展平处理
    size_t K = weight->shape()[1]; // in_features
    size_t N = weight->shape()[0]; // out_features
    size_t M = input->numel() / K; // total batch size

    // 简单检查
    // ASSERT(input->shape().back() == K, "Linear input shape mismatch");

    // 3. 执行矩阵乘法: Out[m, n] = Sum_k (Input[m, k] * Weight[n, k]) + Bias[n]
    // 这是一个朴素的 O(M*N*K) 实现
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            
            float sum = 0.0f;
            
            // 计算点积 (Dot Product)
            for (size_t k = 0; k < K; ++k) {
                // Input offset: m行 k列
                T in_val = in_ptr[m * K + k];
                // Weight offset: n行 k列 (因为 Weight 是 [N, K])
                T w_val = w_ptr[n * K + k];
                
                sum += to_float(in_val) * to_float(w_val);
            }

            // 加上 Bias
            if (b_ptr) {
                sum += to_float(b_ptr[n]);
            }

            // 存回结果
            out_ptr[m * N + n] = from_float<T>(sum);
        }
    }
}

void linear_cpu(tensor_t out, tensor_t input, tensor_t weight, tensor_t bias) {
    auto dtype = input->dtype();

    if (dtype == LLAISYS_DTYPE_F32) {
        linear_kernel<float>(out, input, weight, bias);
    } 
    else if (dtype == LLAISYS_DTYPE_F16) {
        linear_kernel<fp16_t>(out, input, weight, bias);
    } 
    else if (dtype == LLAISYS_DTYPE_BF16) {
        linear_kernel<bf16_t>(out, input, weight, bias);
    } 
    else {
        throw std::runtime_error("Linear: Unsupported data type.");
    }
}

} // namespace llaisys::ops::cpu