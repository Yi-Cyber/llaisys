#include "self_attention_cpu.hpp"
#include <cmath> // exp, max
#include <vector>
#include <limits> // numeric_limits
#include <algorithm> // std::max
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
void self_attention_kernel(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 1. 获取指针
    T* out_ptr = reinterpret_cast<T*>(attn_val->data());
    const T* q_ptr = reinterpret_cast<const T*>(q->data());
    const T* k_ptr = reinterpret_cast<const T*>(k->data());
    const T* v_ptr = reinterpret_cast<const T*>(v->data());

    // 2. 解析维度
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];       // head_dim (Q, K)
    
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];      // head_dim (V, Out)

    size_t group_size = nhead / nkvhead; // GQA 分组大小

    // 预分配 buffer 存储 scores (一行注意力分数)
    std::vector<float> scores(total_len);

    // 3. 遍历 Query (seqlen * nhead)
    for (size_t i = 0; i < seqlen; ++i) {
        // 当前 query token 在完整序列中的绝对位置
        // 假设 Q 是序列的末尾部分
        size_t current_pos = total_len - seqlen + i;

        for (size_t h = 0; h < nhead; ++h) {
            // GQA: 映射到对应的 KV head
            size_t h_kv = h / group_size;

            // 定位 Q 向量: [i, h, :]
            const T* q_vec = q_ptr + (i * nhead + h) * d;

            // --- Step 1: 计算 Attention Scores (Q * K^T) ---
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t j = 0; j < total_len; ++j) {
                // Causal Masking: 如果 KV 位置 j 大于当前 Q 位置，则掩盖
                if (j > current_pos) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                // 定位 K 向量: [j, h_kv, :]
                const T* k_vec = k_ptr + (j * nkvhead + h_kv) * d;

                // 点积
                float dot = 0.0f;
                for (size_t elem = 0; elem < d; ++elem) {
                    dot += to_float(q_vec[elem]) * to_float(k_vec[elem]);
                }
                
                // 缩放
                float score = dot * scale;
                scores[j] = score;
                
                // 记录最大值用于 Softmax
                if (score > max_score) max_score = score;
            }

            // --- Step 2: Softmax ---
            float sum_exp = 0.0f;
            for (size_t j = 0; j <= current_pos; ++j) { // 只遍历有效的 j
                float exp_val = std::exp(scores[j] - max_score);
                scores[j] = exp_val;
                sum_exp += exp_val;
            }
            // 归一化 (不需要显式存回 scores，直接在下一步用)
            float inv_sum_exp = 1.0f / sum_exp;

            // --- Step 3: 加权求和 (Scores * V) ---
            // 定位输出向量: [i, h, :]
            T* out_vec = out_ptr + (i * nhead + h) * dv;
            
            // 初始化输出为 0
            // 注意：这里需要一个 float 的临时 buffer 来累加，最后转回 T
            std::vector<float> acc_out(dv, 0.0f);

            for (size_t j = 0; j <= current_pos; ++j) {
                float weight = scores[j] * inv_sum_exp;
                
                // 定位 V 向量: [j, h_kv, :]
                const T* v_vec = v_ptr + (j * nkvhead + h_kv) * dv;

                for (size_t elem = 0; elem < dv; ++elem) {
                    acc_out[elem] += weight * to_float(v_vec[elem]);
                }
            }

            // 存回结果
            for (size_t elem = 0; elem < dv; ++elem) {
                out_vec[elem] = from_float<T>(acc_out[elem]);
            }
        }
    }
}

void self_attention_cpu(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    auto dtype = q->dtype();

    if (dtype == LLAISYS_DTYPE_F32) {
        self_attention_kernel<float>(attn_val, q, k, v, scale);
    } 
    else if (dtype == LLAISYS_DTYPE_F16) {
        self_attention_kernel<fp16_t>(attn_val, q, k, v, scale);
    } 
    else if (dtype == LLAISYS_DTYPE_BF16) {
        self_attention_kernel<bf16_t>(attn_val, q, k, v, scale);
    } 
    else {
        throw std::runtime_error("Self Attention: Unsupported data type.");
    }
}

} // namespace llaisys::ops::cpu