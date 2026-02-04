#include "rearrange_cpu.hpp"
#include <cstring> // std::memcpy
#include <vector>
#include <stdexcept>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {

namespace {
    // 递归函数：处理任意维度的 Strided Copy
    void rearrange_recursive(
        const std::byte* src_ptr, 
        std::byte* dst_ptr,
        const std::vector<size_t>& shape,
        const std::vector<ptrdiff_t>& src_strides,
        const std::vector<ptrdiff_t>& dst_strides,
        size_t dim, 
        size_t elem_size
    ) {
        // 边界条件：维度耗尽（理论上在 dim == ndim-1 时就处理了，这里是防御性编程）
        if (dim >= shape.size()) return;

        // 当前维度的元素数量
        size_t len = shape[dim];
        
        // 计算当前维度的字节步长
        ptrdiff_t s_stride_bytes = src_strides[dim] * elem_size;
        ptrdiff_t d_stride_bytes = dst_strides[dim] * elem_size;

        if (dim == shape.size() - 1) {
            // --- Base Case: 最内层维度 ---
            for (size_t i = 0; i < len; ++i) {
                // 逐元素拷贝 (因为最后维度可能也不连续，如切片)
                // 如果是连续的，可以用 memcpy 优化整行，但为了通用性，这里逐个拷贝
                std::memcpy(dst_ptr + i * d_stride_bytes, 
                            src_ptr + i * s_stride_bytes, 
                            elem_size);
            }
        } else {
            // --- Recursive Step: 递归处理下一维 ---
            for (size_t i = 0; i < len; ++i) {
                rearrange_recursive(
                    src_ptr + i * s_stride_bytes,
                    dst_ptr + i * d_stride_bytes,
                    shape, src_strides, dst_strides, 
                    dim + 1, elem_size
                );
            }
        }
    }
}

void rearrange_cpu(tensor_t out, tensor_t in) {
    // 0. 特例检查
    if (in->numel() == 0) return;

    // 1. 获取基础信息
    size_t elem_size = in->elementSize();
    const auto& shape = in->shape();
    const auto& src_strides = in->strides();
    const auto& dst_strides = out->strides();
    
    // 2. 获取原始字节指针
    // 注意：const_cast 是因为 data() 有重载，但这里 input 应该是 const 语义
    const std::byte* src_data = in->data();
    std::byte* dst_data = out->data();

    // 3. 处理 0 维张量 (Scalar)
    if (shape.empty()) {
        std::memcpy(dst_data, src_data, elem_size);
        return;
    }

    // 4. 执行递归拷贝
    // 这种方法不依赖具体数据类型 (dtype)，只依赖 elementSize，所以非常通用
    rearrange_recursive(
        src_data, dst_data,
        shape, src_strides, dst_strides,
        0, elem_size
    );
}

} // namespace llaisys::ops::cpu