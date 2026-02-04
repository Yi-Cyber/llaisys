#include "rearrange_cuda.cuh"
#include "../../utils.hpp" 

namespace llaisys::ops::cuda {

void rearrange_cuda(tensor_t out, tensor_t in) {
    // Rearrange 的 CUDA 实现比较复杂（需要自定义 Kernel 处理多维索引），暂时留空
    TO_BE_IMPLEMENTED();
}

} // namespace llaisys::ops::cuda