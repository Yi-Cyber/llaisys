import ctypes
from . import LIB_LLAISYS

# 1. 映射 C++ 的 Meta 结构体
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]

# 2. 映射 C++ 的 Weights 结构体
class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", ctypes.c_void_p),
        ("out_embed", ctypes.c_void_p),
        ("out_norm_w", ctypes.c_void_p),
        # 下面这些是数组指针，指向每一层的权重
        ("attn_norm_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_q_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_q_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_k_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_k_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_v_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_v_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_o_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_norm_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_gate_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_up_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_down_w", ctypes.POINTER(ctypes.c_void_p)),
    ]

# 3. 绑定 C++ 函数
llaisysQwen2ModelCreate = LIB_LLAISYS.llaisysQwen2ModelCreate
llaisysQwen2ModelCreate.argtypes = [
    ctypes.POINTER(LlaisysQwen2Meta),
    ctypes.c_int,           # device_type
    ctypes.POINTER(ctypes.c_int), # device_ids
    ctypes.c_int            # ndevice
]
llaisysQwen2ModelCreate.restype = ctypes.c_void_p

llaisysQwen2ModelDestroy = LIB_LLAISYS.llaisysQwen2ModelDestroy
llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]

llaisysQwen2ModelWeights = LIB_LLAISYS.llaisysQwen2ModelWeights
llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

llaisysQwen2ModelInfer = LIB_LLAISYS.llaisysQwen2ModelInfer
llaisysQwen2ModelInfer.argtypes = [
    ctypes.c_void_p,                # model
    ctypes.POINTER(ctypes.c_int64), # token_ids
    ctypes.c_size_t                 # ntoken
]
llaisysQwen2ModelInfer.restype = ctypes.c_int64