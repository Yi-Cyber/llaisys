#include "embedding_cpu.hpp"
#include <cstring> // for std::memcpy
#include <stdexcept>
#include "../../../tensor/tensor.hpp"

// 包含 core 以获取 fp16_t/bf16_t 定义 (通常在 utils 或 core 中)
// 如果编译报错找不到 fp16_t，请尝试包含 "../../utils.hpp"
#include "../../../core/llaisys_core.hpp" 

namespace llaisys::ops::cpu {

template <typename T>
void embedding_kernel(tensor_t out, tensor_t index, tensor_t weight) {
    // 1. 获取指针 (必须用 reinterpret_cast，因为 data() 返回 std::byte*)
    int64_t* index_ptr = reinterpret_cast<int64_t*>(index->data());
    T* weight_ptr = reinterpret_cast<T*>(weight->data());
    T* out_ptr = reinterpret_cast<T*>(out->data());

    // 2. 获取维度
    size_t num_indices = index->numel();
    size_t hidden_dim = weight->shape()[1];
    
    // 3. 执行查表拷贝
    for (size_t i = 0; i < num_indices; ++i) {
        int64_t idx = index_ptr[i];
        
        T* src = weight_ptr + idx * hidden_dim;
        T* dst = out_ptr + i * hidden_dim;
        
        std::memcpy(dst, src, hidden_dim * sizeof(T));
    }
}

void embedding_cpu(tensor_t out, tensor_t index, tensor_t weight) {
    auto dtype = weight->dtype();

    // 【修正】使用正确的宏定义
    if (dtype == LLAISYS_DTYPE_F32) {
        embedding_kernel<float>(out, index, weight);
    } 
    else if (dtype == LLAISYS_DTYPE_F16) {
        // 尝试支持 F16，如果报错 'fp16_t' 未定义，请暂时注释掉下面一行
        embedding_kernel<fp16_t>(out, index, weight); 
    } 
    else if (dtype == LLAISYS_DTYPE_BF16) {
        // 尝试支持 BF16，如果报错 'bf16_t' 未定义，请暂时注释掉下面一行
        embedding_kernel<bf16_t>(out, index, weight);
    } 
    else {
        throw std::runtime_error("Embedding: Unsupported data type.");
    }
}

} // namespace llaisys::ops::cpu