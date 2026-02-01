#include "argmax_cpu.hpp"
#include "../../../utils.hpp" // 包含 cast 工具
#include <cmath>
#include <cstdint> // int64_t

// 模板函数：实际执行查找逻辑
template <typename T>
void argmax_(int64_t* out_idx, T* out_val, const T* vals, size_t numel) {
    if (numel == 0) return; // 边界情况

    // 初始化：假设第一个元素是最大的
    int64_t best_idx = 0;
    
    // 使用 float 进行比较，防止半精度的精度问题
    // cast 是 utils.hpp 里提供的辅助函数
    float max_v_f = llaisys::utils::cast<float>(vals[0]);
    T max_v_raw = vals[0];

    for (size_t i = 1; i < numel; i++) {
        float curr_v_f = llaisys::utils::cast<float>(vals[i]);
        
        // 找到更大的值，更新索引和值
        if (curr_v_f > max_v_f) {
            max_v_f = curr_v_f;
            max_v_raw = vals[i];
            best_idx = i;
        }
    }

    // 将结果写入输出内存
    *out_idx = best_idx;
    *out_val = max_v_raw;
}

namespace llaisys::ops::cpu {

void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, llaisysDataType_t dtype, size_t numel) {
    // 假设索引输出总是 int64_t 类型
    int64_t* idx_ptr = reinterpret_cast<int64_t*>(max_idx);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_<float>(
            idx_ptr,
            reinterpret_cast<float*>(max_val),
            reinterpret_cast<const float*>(vals),
            numel
        );
    case LLAISYS_DTYPE_BF16:
        return argmax_<llaisys::bf16_t>(
            idx_ptr,
            reinterpret_cast<llaisys::bf16_t*>(max_val),
            reinterpret_cast<const llaisys::bf16_t*>(vals),
            numel
        );
    case LLAISYS_DTYPE_F16:
        return argmax_<llaisys::fp16_t>(
            idx_ptr,
            reinterpret_cast<llaisys::fp16_t*>(max_val),
            reinterpret_cast<const llaisys::fp16_t*>(vals),
            numel
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu