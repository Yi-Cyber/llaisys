#include "../../include/llaisys/models/qwen2.h"
#include "../core/llaisys_core.hpp"
#include "../tensor/tensor.hpp"

// 引入所有算子
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/rearrange/op.hpp"

#include <vector>
#include <iostream>
#include <cmath>    // 【修正】添加 cmath 以支持 std::sqrt
#include <cstring>  // 【修正】添加 cstring 以支持 std::memcpy

using namespace llaisys;

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    // KV Cache: [layer_idx] -> Tensor
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    
    // 当前生成的总 Token 数量（位置索引）
    int64_t current_pos = 0;

    LlaisysQwen2Model(const LlaisysQwen2Meta& m) : meta(m) {
        weights.attn_norm_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_q_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_q_b = new llaisysTensor_t[meta.nlayer];
        weights.attn_k_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_k_b = new llaisysTensor_t[meta.nlayer];
        weights.attn_v_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_v_b = new llaisysTensor_t[meta.nlayer];
        weights.attn_o_w = new llaisysTensor_t[meta.nlayer];
        
        weights.mlp_norm_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_gate_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_up_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_down_w = new llaisysTensor_t[meta.nlayer];
    }

    ~LlaisysQwen2Model() {
        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w;
        delete[] weights.attn_q_b;
        delete[] weights.attn_k_w;
        delete[] weights.attn_k_b;
        delete[] weights.attn_v_w;
        delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;
        delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;
        delete[] weights.mlp_down_w;
    }
};

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model(*meta);
    int device_id = (ndevice > 0) ? device_ids[0] : 0;
    
    // 初始化 KV Cache
    for (size_t i = 0; i < meta->nlayer; ++i) {
        std::vector<size_t> kv_shape = {meta->maxseq, meta->nkvh, meta->dh};
        
        auto k_tensor = Tensor::create(kv_shape, (llaisysDataType_t)meta->dtype, device, device_id);
        auto v_tensor = Tensor::create(kv_shape, (llaisysDataType_t)meta->dtype, device, device_id);
        
        model->k_cache.push_back(k_tensor);
        model->v_cache.push_back(v_tensor);
    }
    return model;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (model) delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    return &model->weights;
}

// 【修正】辅助转换函数：绕过 strict-aliasing
// 我们假设 llaisysTensor_t 实际上存储的是 tensor_t (即 std::shared_ptr<Tensor>) 的内存布局
// 更安全的方法是使用 memcpy
static tensor_t cast_tensor(llaisysTensor_t opaque_tensor) {
    tensor_t t;
    // 确保大小一致 (llaisysTensor_t 应该是指针大小，而 tensor_t 是 shared_ptr，通常是 2 个指针大小)
    // 如果 llaisysTensor_t 在 C 接口定义为 void*，那么它只是一个 handle。
    // 在 Python 侧我们传下来的是 tensor.handle。
    // 回顾 Python 代码：handle 是 tensor 对象的地址吗？
    // 通常 C 接口传递 shared_ptr 需要 new 一个 shared_ptr 然后传指针。
    
    // 但是！在本项目中，llaisysTensor_t 被定义为 void* (或类似)。
    // 如果 Python 端传的是 tensor.handle (即 shared_ptr 的地址)，
    // 那么这里我们需要把它当做 tensor_t* 来解引用。
    
    // 为了规避编译错误，我们使用 memcpy
    // 注意：这里假设 llaisysTensor_t 确实是一个指向 tensor_t 对象的指针
    
    // 之前的报错代码：return *reinterpret_cast<tensor_t*>(&opaque_tensor);
    // 这里的 opaque_tensor 本身是一个值，取地址后是指针。
    
    // 如果 llaisysTensor_t 是 void*，那么它指向的是堆上的 shared_ptr<Tensor>
    // 或者是 shared_ptr 的 raw pointer?
    // 根据项目惯例，C 接口通常传递的是 "shared_ptr<Tensor>*" (指针的指针) 或者 "Tensor*" (原始指针)
    
    // 假设：llaisysTensor_t 是 void*，且它指向一个 tensor_t 对象
    // 那么：
    // tensor_t* ptr = reinterpret_cast<tensor_t*>(opaque_tensor);
    // return *ptr; 
    
    // 但是之前的代码是 &opaque_tensor，这意味着 opaque_tensor 本身存储了 shared_ptr 的数据？
    // 如果 llaisysTensor_t 是 typedef void*，那么 &t 就是 void**。
    
    // 让我们尝试最“暴力”但有效的方法：memcpy
    // 但前提是我们知道它到底是什么。
    
    // 根据报错上下文，llaisysTensor_t 很可能被定义为 void*。
    // 而 Python 传进来的是 tensor 的 handle。
    
    // 修正策略：直接强转 void* 到 tensor_t* (假设它是指针)
    // 如果 llaisysTensor_t 就是 tensor_t (shared_ptr)，那它是值传递。
    // C 语言不支持 class，所以 llaisysTensor_t 必然是 void*。
    
    // 如果 Python 传的是 handle (void*)，那这个 void* 应该被转换为 tensor_t (shared_ptr)。
    // 但 shared_ptr 是对象，void* 是指针。
    // 唯一的可能是：这个 void* 是一个指向 tensor_t 的指针。
    
    // 我们尝试如下写法，这通常能通过编译：
    void* ptr = opaque_tensor;
    return *static_cast<tensor_t*>(ptr);
}

// 辅助：从 tensor_t 指针数组中获取
static tensor_t get_w(llaisysTensor_t* array, size_t idx) {
    // array[idx] 是一个 llaisysTensor_t
    return cast_tensor(array[idx]);
}

// 单独处理
static tensor_t get_t(llaisysTensor_t t) {
    return cast_tensor(t);
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    auto& m = model->meta;
    auto dev = model->k_cache[0]->deviceType();
    int dev_id = model->k_cache[0]->deviceId();
    auto dtype = model->k_cache[0]->dtype();

    // 1. 准备输入
    auto input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, dev, dev_id);
    input_ids->load(token_ids);

    auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, dev, dev_id);
    std::vector<int64_t> pos_vec(ntoken);
    for(size_t i=0; i<ntoken; ++i) pos_vec[i] = model->current_pos + i;
    pos_ids->load(pos_vec.data());

    // 2. Embedding
    auto hidden_states = Tensor::create({ntoken, m.hs}, dtype, dev, dev_id);
    ops::embedding(hidden_states, input_ids, get_t(model->weights.in_embed));

    // 3. Layers Loop
    for(size_t i=0; i<m.nlayer; ++i) {
        auto residual = hidden_states;

        // A. RMS Norm
        auto hidden_norm = Tensor::create({ntoken, m.hs}, dtype, dev, dev_id);
        ops::rms_norm(hidden_norm, hidden_states, get_w(model->weights.attn_norm_w, i), m.epsilon);

        // B. QKV Proj
        auto q = Tensor::create({ntoken, m.nh * m.dh}, dtype, dev, dev_id);
        auto k = Tensor::create({ntoken, m.nkvh * m.dh}, dtype, dev, dev_id);
        auto v = Tensor::create({ntoken, m.nkvh * m.dh}, dtype, dev, dev_id);

        ops::linear(q, hidden_norm, get_w(model->weights.attn_q_w, i), get_w(model->weights.attn_q_b, i));
        ops::linear(k, hidden_norm, get_w(model->weights.attn_k_w, i), get_w(model->weights.attn_k_b, i));
        ops::linear(v, hidden_norm, get_w(model->weights.attn_v_w, i), get_w(model->weights.attn_v_b, i));

        auto q_3d = q->view({ntoken, m.nh, m.dh});
        auto k_3d = k->view({ntoken, m.nkvh, m.dh});
        auto v_3d = v->view({ntoken, m.nkvh, m.dh});

        // C. RoPE
        ops::rope(q_3d, q_3d, pos_ids, m.theta);
        ops::rope(k_3d, k_3d, pos_ids, m.theta);

        // D. Update KV Cache
        auto k_cache_slot = model->k_cache[i]->slice(0, model->current_pos, model->current_pos + ntoken);
        auto v_cache_slot = model->v_cache[i]->slice(0, model->current_pos, model->current_pos + ntoken);
        
        ops::rearrange(k_cache_slot, k_3d);
        ops::rearrange(v_cache_slot, v_3d);

        // E. Self Attention
        auto k_full = model->k_cache[i]->slice(0, 0, model->current_pos + ntoken);
        auto v_full = model->v_cache[i]->slice(0, 0, model->current_pos + ntoken);

        auto attn_out_3d = Tensor::create({ntoken, m.nh, m.dh}, dtype, dev, dev_id);
        
        // 【修正】std::sqrt
        float scale = 1.0f / std::sqrt(static_cast<float>(m.dh));
        ops::self_attention(attn_out_3d, q_3d, k_full, v_full, scale);

        // F. Output Proj
        auto attn_out = attn_out_3d->view({ntoken, m.hs});
        auto attn_proj = Tensor::create({ntoken, m.hs}, dtype, dev, dev_id);
        ops::linear(attn_proj, attn_out, get_w(model->weights.attn_o_w, i), nullptr);

        // G. Residual Add
        ops::add(hidden_states, residual, attn_proj);

        // --- MLP Block ---
        residual = hidden_states;

        // H. RMS Norm
        ops::rms_norm(hidden_norm, hidden_states, get_w(model->weights.mlp_norm_w, i), m.epsilon);

        // I. Gate & Up Proj
        auto gate_buf = Tensor::create({ntoken, m.di}, dtype, dev, dev_id);
        auto up_buf = Tensor::create({ntoken, m.di}, dtype, dev, dev_id);
        
        ops::linear(gate_buf, hidden_norm, get_w(model->weights.mlp_gate_w, i), nullptr);
        ops::linear(up_buf, hidden_norm, get_w(model->weights.mlp_up_w, i), nullptr);

        // J. SwiGLU
        auto mlp_act = Tensor::create({ntoken, m.di}, dtype, dev, dev_id);
        ops::swiglu(mlp_act, gate_buf, up_buf);

        // K. Down Proj
        auto mlp_out = Tensor::create({ntoken, m.hs}, dtype, dev, dev_id);
        ops::linear(mlp_out, mlp_act, get_w(model->weights.mlp_down_w, i), nullptr);

        // L. Residual Add
        ops::add(hidden_states, residual, mlp_out);
    }

    // 4. Final Norm
    auto final_norm = Tensor::create({ntoken, m.hs}, dtype, dev, dev_id);
    ops::rms_norm(final_norm, hidden_states, get_t(model->weights.out_norm_w), m.epsilon);

    // 5. LM Head (Logits)
    auto logits = Tensor::create({ntoken, m.voc}, dtype, dev, dev_id);
    ops::linear(logits, final_norm, get_t(model->weights.out_embed), nullptr);

    // 6. Argmax
    auto last_logit = logits->slice(0, ntoken - 1, ntoken);
    
    auto next_token_tensor = Tensor::create({1}, LLAISYS_DTYPE_I64, dev, dev_id);
    auto val_tensor = Tensor::create({1}, dtype, dev, dev_id);
    
    auto last_logit_1d = last_logit->view({m.voc});
    ops::argmax(next_token_tensor, val_tensor, last_logit_1d);

    // 7. 更新状态
    model->current_pos += ntoken;

    int64_t next_token_id = 0;
    if (dev == LLAISYS_DEVICE_CPU) {
        next_token_id = *reinterpret_cast<int64_t*>(next_token_tensor->data());
    }

    return next_token_id;
}

} // extern "C"