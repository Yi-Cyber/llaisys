#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>
#include <algorithm> // for std::copy
#include <stdexcept>

namespace llaisys {

// ==========================================
// 辅助函数：用于处理非连续内存的拷贝
// ==========================================
namespace {
    // 递归将 Strided 数据拷贝到 连续内存中
    void copy_strided_recursive(const std::byte* src, std::byte* dst, 
                                const std::vector<size_t>& shape, 
                                const std::vector<ptrdiff_t>& strides, 
                                size_t dim, size_t elem_size) {
        if (dim == shape.size() - 1) {
            // 最内层维度
            for (size_t i = 0; i < shape[dim]; ++i) {
                std::memcpy(dst + i * elem_size, src + i * strides[dim], elem_size);
            }
        } else {
            // 递归外层
            size_t inner_block_size = 1;
            for(size_t k=dim+1; k<shape.size(); ++k) inner_block_size *= shape[k];
            inner_block_size *= elem_size;

            for (size_t i = 0; i < shape[dim]; ++i) {
                copy_strided_recursive(src + i * strides[dim], 
                                       dst + i * inner_block_size, 
                                       shape, strides, dim + 1, elem_size);
            }
        }
    }
}

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    if (_meta.shape.empty()) return 0;
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype() 
       << " device=" << (this->deviceType() == LLAISYS_DEVICE_CPU ? "CPU" : "GPU");

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (shape.empty()) return; 
    
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        std::cout << "[" << std::endl;
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
        std::cout << "]" << std::endl;
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    
    // 如果是 GPU 且不连续，为了 debug 方便，这里不通过 complex copy，而是直接 dump
    // 注意：这里复用了 contiguous() 逻辑可能会更好，但为了 debug 独立性，保留原逻辑略微修改
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        // GPU case: Copy to host first
        // 计算实际需要的连续 buffer 大小 (这是一个简单的全量 copy，如果 slice 很小但 original很大，会比较浪费，但在 debug 无所谓)
        // 更好的做法是先 contiguous 再 D2H
        auto contig_tensor = this->contiguous(); // 确保它是连续的
        auto tmp_tensor = create(contig_tensor->shape(), contig_tensor->dtype(), LLAISYS_DEVICE_CPU);
        
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            contig_tensor->data(),
            contig_tensor->numel() * contig_tensor->elementSize(),
            LLAISYS_MEMCPY_D2H);
            
        debug_print(tmp_tensor->data(), tmp_tensor->shape(), tmp_tensor->strides(), tmp_tensor->dtype());
    }
}

// ==========================================
// 核心实现补全开始
// ==========================================

bool Tensor::isContiguous() const {
    size_t z = 1;
    // 从最后一个维度向前遍历
    // 最后一个维度的 stride 必须是 1
    // 第 i 个维度的 stride 必须等于 shape[i+1] * stride[i+1]
    for (int i = _meta.shape.size() - 1; i >= 0; i--) {
        if (_meta.shape[i] != 1) { // 维度为1不影响连续性判断
            if (_meta.strides[i] != static_cast<ptrdiff_t>(z)) {
                return false;
            }
            z *= _meta.shape[i];
        }
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != _meta.shape.size()) {
        throw std::runtime_error("Permute dimension mismatch");
    }

    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());

    for (size_t i = 0; i < order.size(); ++i) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    // 共享 Storage，只改变 Meta
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    if (!isContiguous()) {
        throw std::runtime_error("View can only be called on contiguous tensors. Use reshape() instead.");
    }

    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        throw std::runtime_error("Shape mismatch in view()");
    }

    // 计算新的连续 strides
    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= shape[i];
    }

    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= _meta.shape.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    
    // Clamp start/end
    size_t dim_size = _meta.shape[dim];
    if (start > dim_size) start = dim_size;
    if (end > dim_size) end = dim_size;
    if (start >= end) {
         // 返回空张量逻辑，这里简化处理，假设用户输入有效
         // 实际工程中可能需要处理 0-size tensor
    }

    size_t len = end - start;
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = len;

    // Strides 不变，offset 增加
    size_t new_offset = _offset + start * _meta.strides[dim] * elementSize(); // offset is in bytes

    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    if (!isContiguous()) {
        throw std::runtime_error("Loading data into non-contiguous tensor is not supported directly. Make it contiguous first.");
    }

    size_t size_bytes = numel() * elementSize();
    core::context().setDevice(this->deviceType(), this->deviceId());
    
    // 假设 src_ 位于 Host 内存
    core::context().runtime().api()->memcpy_sync(
        this->data(), 
        src_, 
        size_bytes, 
        this->deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D
    );
}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        // 如果已经连续，通常返回自身的副本或直接返回自身
        // 这里为了安全语义，返回一个新的 Tensor 对象，但共享内存？
        // 不，contiguous() 语义通常意味着获得一段独立的连续内存。
        // 如果已经是连续的，我们可以做一个深拷贝(Deep Copy)或者返回自身。
        // PyTorch behavior: if contiguous, returns self. Let's create a copy to be safe or just return shallow copy.
        // 为了实现简单且符合 C++ RAII，我们做一个深拷贝，确保返回的是全新的连续内存块
        
        auto new_tensor = Tensor::create(_meta.shape, _meta.dtype, deviceType(), deviceId());
        
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(
            new_tensor->data(),
            this->data(),
            numel() * elementSize(),
            deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D
        );
        return new_tensor;
    }

    // 如果不连续，需要分配新内存并重排数据
    auto new_tensor = Tensor::create(_meta.shape, _meta.dtype, deviceType(), deviceId());
    
    if (deviceType() == LLAISYS_DEVICE_CPU) {
        copy_strided_recursive(this->data(), new_tensor->data(), 
                               _meta.shape, _meta.strides, 0, elementSize());
    } else {
        // GPU Case: 复杂。因为没有 kernel，我们需要：
        // 1. 把当前非连续数据 copy 到 CPU (需要逐个元素或者整个storage copy下来然后在CPU切)
        // 为了通用性，我们这里把整个 Storage 拷贝回 CPU，在 CPU 做重排，然后再上传。
        // 这非常慢，但是是目前架构下唯一不需要写 CUDA Kernel 的做法。
        
        // 1. 创建 CPU 端的临时 Tensor (Source)
        size_t host_storage_size = _storage->size(); // Bytes
        auto host_storage = core::context().runtime().allocateHostStorage(host_storage_size);
        
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(
            host_storage->memory(),
            _storage->memory(),
            host_storage_size,
            LLAISYS_MEMCPY_D2H
        );
        
        // 构建一个 CPU 上的 View 指向这个 host storage
        TensorMeta host_meta = _meta; 
        // 使用私有构造，offset 保持一致
        // Hack: 我们不能直接调用私有构造，但我们可以手动模拟那个过程，或者这里假设我们有一个 helper
        // 由于无法直接构造 Tensor 对象(私有构造)，我们手动调用 recursive copy
        
        // 2. 创建 CPU 端的 Destination (Contiguous)
        size_t total_bytes = numel() * elementSize();
        std::vector<std::byte> host_dst(total_bytes);

        // 3. 在 CPU 上做 Strided Copy
        copy_strided_recursive(host_storage->memory() + _offset, host_dst.data(), 
                               _meta.shape, _meta.strides, 0, elementSize());

        // 4. 将整理好的连续数据上传到新的 GPU Tensor
        core::context().runtime().api()->memcpy_sync(
            new_tensor->data(),
            host_dst.data(),
            total_bytes,
            LLAISYS_MEMCPY_H2D
        );
    }
    
    return new_tensor;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        throw std::runtime_error("Shape mismatch in reshape()");
    }

    if (isContiguous()) {
        return view(shape);
    } else {
        // 如果不连续，必须先 contiguous() 变成连续内存，再 view
        return contiguous()->view(shape);
    }
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    // 1. 创建目标设备上的 Tensor
    // 注意：to() 操作通常会把 Tensor 变成连续的
    auto new_tensor = Tensor::create(_meta.shape, _meta.dtype, device_type, device);

    // 2. 如果源数据本身不连续，先在源设备上变连续，便于拷贝
    // 或者我们直接复用 contiguous() 里的逻辑。最简单的方法是：
    tensor_t src_contiguous;
    if (isContiguous()) {
        // 这里只是为了获取一个 shared_ptr，实际上并不拷贝数据，因为 const this 无法直接转 shared_ptr
        // 我们利用一个浅拷贝 view 来获取 shared_ptr wrapper
        src_contiguous = std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    } else {
        src_contiguous = this->contiguous();
    }

    // 3. 执行拷贝
    core::context().setDevice(device_type, device); // 设置目标设备上下文
    
    // 确定 memcpy 类型
    llaisysMemcpyKind_t kind;
    if (this->deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) kind = LLAISYS_MEMCPY_H2H;
    else if (this->deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) kind = LLAISYS_MEMCPY_H2D;
    else if (this->deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) kind = LLAISYS_MEMCPY_D2H;
    else kind = LLAISYS_MEMCPY_D2D;

    core::context().runtime().api()->memcpy_sync(
        new_tensor->data(),
        src_contiguous->data(),
        numel() * elementSize(),
        kind
    );

    return new_tensor;
}

} // namespace llaisys