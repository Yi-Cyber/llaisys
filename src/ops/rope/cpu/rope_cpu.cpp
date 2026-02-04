#include "rope_cpu.hpp"
#include <cmath> // for std::sin, std::cos, std::pow
#include <stdexcept>
#include "../../../tensor/tensor.hpp"
#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp" 

namespace llaisys::ops::cpu {

// 辅助函数
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
void rope_kernel(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 1. 获取指针
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());

    // 2. 解析维度
    size_t seq_len = in->shape()[0];
    size_t n_head = in->shape()[1];
    size_t head_dim = in->shape()[2]; // d
    
    // 【修正 1】计算半维度，用于切分
    size_t half_dim = head_dim / 2;
    
    size_t stride_seq = n_head * head_dim;
    size_t stride_head = head_dim;

    // 3. 遍历计算
    for (size_t s = 0; s < seq_len; ++s) {
        int64_t p = pos_ptr[s];
        
        for (size_t h = 0; h < n_head; ++h) {
            size_t offset = s * stride_seq + h * stride_head;
            const T* vec_in = in_ptr + offset;
            T* vec_out = out_ptr + offset;

            // 【修正 2】Half-Split 模式循环
            // 遍历前一半维度 k = 0 ... half_dim - 1
            // 配对元素为: vec[k] 和 vec[k + half_dim]
            for (size_t k = 0; k < half_dim; ++k) {
                
                // 读取：x_a 在前，x_b 在后
                float a = to_float(vec_in[k]);             // x[..., :d/2]
                float b = to_float(vec_in[k + half_dim]);  // x[..., d/2:]

                // 计算角度 phi
                // Python公式: freqs = positions / (theta ** (2 * i / head_dim))
                // 这里的 k 对应 python 中的 i
                float exp_val = (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim);
                float freq = static_cast<float>(p) / std::pow(theta, exp_val);
                
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                // 执行旋转公式 (参考 Python 脚本)
                // y[..., :d/2] = x_a * cos - x_b * sin
                // y[..., d/2:] = x_b * cos + x_a * sin
                float out_a = a * cos_val - b * sin_val;
                float out_b = b * cos_val + a * sin_val;

                // 写回结果：分别写回前半部分和后半部分
                vec_out[k]            = from_float<T>(out_a);
                vec_out[k + half_dim] = from_float<T>(out_b);
            }
        }
    }
}

void rope_cpu(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    auto dtype = in->dtype();

    if (dtype == LLAISYS_DTYPE_F32) {
        rope_kernel<float>(out, in, pos_ids, theta);
    } 
    else if (dtype == LLAISYS_DTYPE_F16) {
        rope_kernel<fp16_t>(out, in, pos_ids, theta);
    } 
    else if (dtype == LLAISYS_DTYPE_BF16) {
        rope_kernel<bf16_t>(out, in, pos_ids, theta);
    } 
    else {
        throw std::runtime_error("RoPE: Unsupported data type.");
    }
}

} // namespace llaisys::ops::cpu