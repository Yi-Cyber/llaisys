#pragma once
#include "../core/llaisys_core.hpp"   // 引入: llaisys_core.hpp  核心库

#include <vector>  // 引入标准库 vector，用于存储形状（shape）和步长（strides）
namespace llaisys {  // 定义在 llaisys 命名空间下，防止符号冲突。
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;

struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
};

class Tensor {
private:
    TensorMeta _meta;   // 包含形状、步长等元数据。
    core::storage_t _storage;  // 底层原始内存块（通常封装了指针和设备分配逻辑）。
    size_t _offset;   // 当前张量视图相对于原始内存块起始位置的偏移量（对 Slice 操作至关重要）。
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);  // 私有构造函数：外部只能通过静态方法 create 来创建张量。

public:
    // 静态工厂方法：指定形状、类型、设备类型（CPU/GPU）和设备 ID 来创建张量。
    static tensor_t create(
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default;  // 默认析构函数。
    // Info
    std::byte *data();
    const std::byte *data() const;
    size_t ndim() const;  // 返回维度的数量。
    const std::vector<size_t> &shape() const;  // 获取元数据，数据的形状
    const std::vector<ptrdiff_t> &strides() const;   // 获取元数据，数据的步长
    llaisysDataType_t dtype() const;    // 获取元数据，数据的类型
    llaisysDeviceType_t deviceType() const;  // 数据所处硬件种类
    int deviceId() const;   // 数据所处硬件ID
    size_t numel() const;   // 获取元素总数（所有维度相乘）。
    size_t elementSize() const;   // 获取单个元素占用的字节数。

    std::string info() const;   // 返回张量的字符串信息
    void debug() const;   // 打印调试信息

    bool isContiguous() const;   // 判断内存是否按照步长顺序连续排列（没有被 permute 或 slice 搞乱顺序）。

    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;  // 维度重排
    tensor_t slice(size_t dim, size_t start, size_t end) const;  // 维度切片
    tensor_t view(const std::vector<size_t> &shape) const;  // 视图变换

    // Load data from host memory
    void load(const void *src);  // 从主机内存加载数据到张量

    // Challenging features
    tensor_t contiguous() const;  // 返回一个内存连续的张量副本
    tensor_t reshape(const std::vector<size_t> &shape) const;   // 返回一个新的张量，形状被重塑
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;   // 张量数据迁移到指定设备
};

} // namespace llaisys
