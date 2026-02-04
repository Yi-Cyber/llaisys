#include "swiglu_cpu.hpp"
#include <cmath> // std::exp
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
void swiglu_kernel(tensor_t out, tensor_t gate, tensor_t up) {
    // 1. 获取指针
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* gate_ptr = reinterpret_cast<const T*>(gate->data());
    const T* up_ptr = reinterpret_cast<const T*>(up->data());

    // 2. 获取元素总数 (逐元素操作，无需关心形状，只看 numel)
    size_t n = out->numel();

    // 3. 遍历计算
    for (size_t i = 0; i < n; ++i) {
        float val_gate = to_float(gate_ptr[i]);
        float val_up = to_float(up_ptr[i]);

        // SiLU(x) = x / (1 + exp(-x))
        float silu = val_gate / (1.0f + std::exp(-val_gate));
        
        // out = up * SiLU(gate)
        float res = val_up * silu;

        out_ptr[i] = from_float<T>(res);
    }
}

void swiglu_cpu(tensor_t out, tensor_t gate, tensor_t up) {
    auto dtype = out->dtype();

    if (dtype == LLAISYS_DTYPE_F32) {
        swiglu_kernel<float>(out, gate, up);
    } 
    else if (dtype == LLAISYS_DTYPE_F16) {
        swiglu_kernel<fp16_t>(out, gate, up);
    } 
    else if (dtype == LLAISYS_DTYPE_BF16) {
        swiglu_kernel<bf16_t>(out, gate, up);
    } 
    else {
        throw std::runtime_error("SwiGLU: Unsupported data type.");
    }
}

} // namespace llaisys::ops::cpu