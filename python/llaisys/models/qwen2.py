from typing import Sequence
import ctypes
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open  # 【修改】使用通用 safe_open
import torch                       # 【新增】引入 torch 用于处理 bf16

from ..libllaisys import DeviceType
from ..tensor import Tensor, DataType
# 导入 Ctypes 接口
from ..libllaisys.qwen2 import (
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
    llaisysQwen2ModelCreate,
    llaisysQwen2ModelDestroy,
    llaisysQwen2ModelWeights,
    llaisysQwen2ModelInfer
)

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # 1. 读取 Config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)

        # 2. 填充 Meta 数据
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = DataType.F32.value # C++ 引擎使用 F32
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["hidden_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = config["intermediate_size"]
        self.meta.maxseq = 2048
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config["rms_norm_eps"]
        self.meta.theta = config.get("rope_theta", 10000.0)
        self.meta.end_token = 151643 # <|endoftext|>

        # 3. 创建 C++ 模型实例
        self.model_ptr = llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            device.value,
            None, 0
        )
        
        c_weights = llaisysQwen2ModelWeights(self.model_ptr).contents
        self.keep_alive_tensors = []

        print(f"Loading weights from {model_path}...")
        
        # 4. 加载权重
        for file in sorted(model_path.glob("*.safetensors")):
            # 【核心修改】framework="pt" 使用 PyTorch 后端读取，支持 bf16
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # 1. 读取为 PyTorch Tensor (支持 bfloat16)
                    pt_tensor = f.get_tensor(key)
                    
                    # 2. 转换为 float32 并导出为 numpy
                    # 这一步解决了 "bfloat16 not understood" 的问题
                    np_data = pt_tensor.to(torch.float32).numpy()
                    
                    # 3. 传给 LLAISYS Tensor
                    # ✅ 新代码：手动构建流程

                    # 1. 确保 numpy 数组在内存中是连续的（非常重要，否则数据会错乱）
                    if not np_data.flags['C_CONTIGUOUS']:
                        np_data = np.ascontiguousarray(np_data)

                    # 2. 使用 shape 初始化 Tensor
                    # 注意：直接在 init 中传入 device，因为我看代码里没有 .to() 方法
                    tensor = Tensor(
                        shape=np_data.shape, 
                        device=device
                        # 如果 np_data 不是 float32，这里可能还需要指定 dtype=...
                    )

                    # 3. 获取 NumPy 数组的 C 指针
                    data_ptr = np_data.ctypes.data_as(ctypes.c_void_p)

                    # 4. 调用底层的 load 方法将数据复制进去
                    tensor.load(data_ptr)
                    self.keep_alive_tensors.append(tensor)
                    # ✅ 正确代码
                    handle = tensor.lib_tensor()

                    # 映射逻辑保持不变
                    if "model.embed_tokens" in key: c_weights.in_embed = handle
                    elif "lm_head" in key: c_weights.out_embed = handle
                    elif "model.norm.weight" in key: c_weights.out_norm_w = handle
                    elif "layers" in key:
                        parts = key.split(".")
                        layer_idx = int(parts[2]) 
                        if "input_layernorm" in key: c_weights.attn_norm_w[layer_idx] = handle
                        elif "post_attention_layernorm" in key: c_weights.mlp_norm_w[layer_idx] = handle
                        elif "self_attn" in key:
                            if "q_proj.weight" in key:   c_weights.attn_q_w[layer_idx] = handle
                            elif "q_proj.bias" in key:   c_weights.attn_q_b[layer_idx] = handle
                            elif "k_proj.weight" in key: c_weights.attn_k_w[layer_idx] = handle
                            elif "k_proj.bias" in key:   c_weights.attn_k_b[layer_idx] = handle
                            elif "v_proj.weight" in key: c_weights.attn_v_w[layer_idx] = handle
                            elif "v_proj.bias" in key:   c_weights.attn_v_b[layer_idx] = handle
                            elif "o_proj.weight" in key: c_weights.attn_o_w[layer_idx] = handle
                        elif "mlp" in key:
                            if "gate_proj" in key: c_weights.mlp_gate_w[layer_idx] = handle
                            elif "up_proj" in key: c_weights.mlp_up_w[layer_idx] = handle
                            elif "down_proj" in key: c_weights.mlp_down_w[layer_idx] = handle
        print("Weights loaded successfully.")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 100,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        tokens = list(inputs)
        
        # --- 阶段 1: Prefill (预填充) ---
        c_inputs = (ctypes.c_int64 * len(tokens))(*tokens)
        
        next_token = llaisysQwen2ModelInfer(
            self.model_ptr,
            c_inputs,
            len(tokens)
        )
        
        if next_token == self.meta.end_token:
            return []
            
        tokens.append(next_token)
        # print(f"Generated: {next_token}", flush=True)
        
        # --- 阶段 2: Decode (增量解码) ---
        for _ in range(max_new_tokens - 1):
            c_input = (ctypes.c_int64 * 1)(*[next_token])
            
            next_token = llaisysQwen2ModelInfer(
                self.model_ptr,
                c_input,
                1 
            )
            
            if next_token == self.meta.end_token:
                break
            
            tokens.append(next_token)

        return tokens[len(inputs):]

    def __del__(self):
        if hasattr(self, "model_ptr") and self.model_ptr:
            llaisysQwen2ModelDestroy(self.model_ptr)