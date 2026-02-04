#include "rms_norm_cpu.hpp"
#include <cmath> // for std::sqrt
#include <stdexcept>
#include "../../../tensor/tensor.hpp"
#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp" 

namespace llaisys::ops::cpu {

// 【经验】辅助函数：确保计算在 float 精度下进行
template<typename T>
float to_float(T val) {
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        return utils::cast<float>(val);
    }
}

template<typename T>
T from_float(float val) {
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        return utils::cast<T>(val);
    }
}

template <typename T>
void rms_norm_kernel(tensor_t out, tensor_t input, tensor_t weight, float eps) {
    // 1. 获取指针 (使用 reinterpret_cast)
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* in_ptr = reinterpret_cast<const T*>(input->data());
    const T* w_ptr = reinterpret_cast<const T*>(weight->data());

    // 2. 解析维度
    size_t last_dim = input->shape().back(); // N (hidden_size)
    size_t num_rows = input->numel() / last_dim; // M (batch_size * seq_len)

    // 3. 逐行计算
    for (size_t i = 0; i < num_rows; ++i) {
        // 定位当前行的指针
        const T* row_in = in_ptr + i * last_dim;
        T* row_out = out_ptr + i * last_dim;

        // --- Step 1: 计算平方和 (Sum of Squares) ---
        float sum_sq = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) {
            float val = to_float(row_in[j]);
            sum_sq += val * val;
        }

        // --- Step 2: 计算 RMS ---
        // Mean Square = sum_sq / d
        // RMS = sqrt(Mean Square + eps)
        float mean_sq = sum_sq / static_cast<float>(last_dim);
        float rms = std::sqrt(mean_sq + eps);
        
        // 预计算倒数，乘法比除法快
        float inv_rms = 1.0f / rms;

        // --- Step 3: 归一化并缩放 (Normalize & Scale) ---
        // Y_i = (X_i / RMS) * W_i
        for (size_t j = 0; j < last_dim; ++j) {
            float val = to_float(row_in[j]);
            float w_val = to_float(w_ptr[j]);
            
            float res = val * inv_rms * w_val;
            
            row_out[j] = from_float<T>(res);
        }
    }
}

void rms_norm_cpu(tensor_t out, tensor_t input, tensor_t weight, float eps) {
    auto dtype = input->dtype();

    // 【经验】使用标准宏定义分发
    if (dtype == LLAISYS_DTYPE_F32) {
        rms_norm_kernel<float>(out, input, weight, eps);
    } 
    else if (dtype == LLAISYS_DTYPE_F16) {
        rms_norm_kernel<fp16_t>(out, input, weight, eps);
    } 
    else if (dtype == LLAISYS_DTYPE_BF16) {
        rms_norm_kernel<bf16_t>(out, input, weight, eps);
    } 
    else {
        throw std::runtime_error("RMS Norm: Unsupported data type.");
    }
}

} // namespace llaisys::ops::cpu