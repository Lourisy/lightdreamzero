import os
import logging
import torch
import torch.nn as nn
from functools import partial

logger = logging.getLogger(__name__)

# Try to import sageattention
try:
    from sageattention import sageattn
    SAGEATTENTION_AVAILABLE = True
except ImportError:
    SAGEATTENTION_AVAILABLE = False


def _sage_attention_forward(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, causal=False, **kwargs):
    """
    Wrapper for SageAttention to match the interface of Wan2.1 flash_attention.
    q, k, v are typically [B, L, H, D] in Wan2.1.
    SageAttention typically expects [B, H, L, D] and provides a very optimized kernel.
    """
    # Wan2.1 AttentionModule passes q,k,v as [B, L, H, D]
    # We transpose to [B, H, L, D] for sageattn
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # SageAttention expects fp16/bf16
    dtype = q.dtype
    if dtype not in [torch.bfloat16, torch.float16]:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        
    out = sageattn(q, k, v, is_causal=causal, sm_scale=softmax_scale)
    
    # Transpose back to [B, L, H, D]
    out = out.transpose(1, 2).contiguous()
    return out.to(dtype)


def apply_sageattention_to_model(model: nn.Module):
    """
    Replaces the ATTENTION_BACKEND of Wan2.1 AttentionModules with SageAttention if available.
    """
    if not SAGEATTENTION_AVAILABLE:
        logger.warning("SageAttention is not installed. Skipping patch. (pip install sageattention)")
        return False
        
    count = 0
    from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule
    for name, module in model.named_modules():
        if isinstance(module, AttentionModule):
            # Patch the backend
            module.backend = "SageAttention"
            module.attn_func = _sage_attention_forward
            count += 1
            
    logger.info(f"Patched {count} AttentionModules with SageAttention.")
    return True


class Fp8Linear(nn.Module):
    """Drop-in nn.Linear replacement using FP8 storage + torch._scaled_mm for
    FP8-accelerated matmul on Hopper (H100) / Blackwell (B200/5090) GPUs.
    - Storage: FP8 e4m3fn weights → ~50% VRAM savings vs BF16
    - Compute: FP8 tensor core matmul → ~2x throughput vs BF16"""

    _FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

    def __init__(self, original: nn.Linear):
        super().__init__()
        w = original.weight.data.float()
        amax = w.abs().amax().clamp(min=1e-12)
        scale = self._FP8_MAX / amax

        # Weight stored in original [out_features, in_features] layout (row-major).
        # At forward time we pass weight.t() which gives a column-major view,
        # satisfying cuBLASLt's requirement: A=row-major × B=column-major.
        self.register_buffer(
            "weight_fp8",
            (w * scale).clamp(-self._FP8_MAX, self._FP8_MAX)
            .to(torch.float8_e4m3fn)
            .contiguous(),
        )
        self.register_buffer(
            "weight_scale_inv",
            torch.tensor(1.0 / scale.item(), dtype=torch.float32),
        )
        self.bias = (
            nn.Parameter(original.bias.data.clone(), requires_grad=False)
            if original.bias is not None
            else None
        )
        self.in_features = original.in_features
        self.out_features = original.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        compute_dtype = x.dtype
        x = x.reshape(-1, self.in_features).contiguous()

        # Dynamic per-tensor quantization of input activations to FP8
        amax = x.abs().amax().clamp(min=1e-12)
        scale = self._FP8_MAX / amax
        x_fp8 = (x * scale).clamp(-self._FP8_MAX, self._FP8_MAX).to(torch.float8_e4m3fn)
        x_scale_inv = (1.0 / scale).to(torch.float32)

        # FP8 tensor core matmul: A[M,K] (row-major) × B[K,N] (column-major via .t())
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),  # column-major view, no copy
            scale_a=x_scale_inv,
            scale_b=self.weight_scale_inv,
            out_dtype=compute_dtype,
            use_fast_accum=True,
        )

        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)


def _set_submodule(root: nn.Module, path: str, new_module: nn.Module):
    """Set a submodule given a dot-separated path (e.g. 'blocks.0.ffn.0')."""
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new_module
    else:
        setattr(parent, parts[-1], new_module)


def apply_fp8_quantization_to_dit(model: nn.Module):
    """
    Applies FP8 conversion to the DiT backbone only, avoiding precision-sensitive
    action heads or VAE decoders. Uses Fp8Linear wrapper so that forward() works
    even when the input is BFloat16 (dynamic dequantization).
    """
    try:
        from torch._dynamo.utils import is_fp8_supported
        # In newer PyTorch versions or with transformer_engine, we can use float8_e4m3fn
        if not hasattr(torch, "float8_e4m3fn"):
            logger.warning("PyTorch version does not support FP8. Skipping FP8 quantization.")
            return False
    except ImportError:
        pass

    count = 0
    # Collect (name, module) pairs first to avoid mutating while iterating
    replacements: list[tuple[str, nn.Linear]] = []

    # Apply to heavy Transformer blocks (T5 and DiT) while completely avoiding sensitive regression heads
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 1. Skip ALL sensitive action decoders, adapters, and final output heads
            sensitive_keywords = [
                "proj_out",              # Action regressions
                "control_adapter",       # Joint condition
                "action_head.model.head",# DiT final output projection
                "img_emb",               # Vision condition
                "time_",                 # Time embeddings
                "patch_"                 # Patch embeddings
            ]
            if any(k in name for k in sensitive_keywords):
                continue
                
            # 2. Skip any other components in the action_head outside of the main DiT '.model'
            if "action_head" in name and "action_head.model." not in name:
                continue

            # If it passes the above, it's a safe heavy linear layer (DiT blocks or T5 textual blocks)
            if getattr(module, 'weight', None) is not None:
                replacements.append((name, module))

    # Now perform the actual replacements
    for name, module in replacements:
        try:
            fp8_module = Fp8Linear(module)
            _set_submodule(model, name, fp8_module)
            count += 1
        except Exception as e:
            logger.debug(f"Could not convert module {name} to fp8: {e}")
                    
    logger.info(f"Replaced {count} Heavy Linear layers (DiT + T5) with Fp8Linear to save VRAM.")


def apply_lightx2v_optimizations(
    model: nn.Module, 
    use_sageattention: bool = False, 
    use_fp8: bool = False,
    compile_model: bool = False
):
    """
    Entry point for LightX2V-style optimizations on the DreamZero model.
    """
    logger.info("Applying LightX2V optimizations...")
    
    if use_sageattention:
        apply_sageattention_to_model(model)
        
    if use_fp8:
        # Check if the device is Ampere or newer (FP8 hardware support)
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 9: # Hopper, Blackwell
                apply_fp8_quantization_to_dit(model)
            else:
                logger.warning(f"FP8 is only supported natively on Hopper (H100) or newer GPUs (Capability >= 9.0). Detected A100/Ampere (Capability {cap}). Skipping FP8 to avoid emulation slowdown.")
        else:
            apply_fp8_quantization_to_dit(model)
            
    if compile_model:
        # Dynamically compile the backbone using Triton/inductor
        logger.info("Setting up torch.compile for DiT backbone...")
        # Reduce compilation time by limiting ops
        torch._dynamo.config.suppress_errors = True
        # In DreamZero, the DiT model is inside model.action_head.model (WanModel)
        if hasattr(model, "action_head") and hasattr(model.action_head, "model"):
            model.action_head.model = torch.compile(
                model.action_head.model, 
                mode="reduce-overhead",
                fullgraph=False
            )
            logger.info("Successfully wrapped DiT backbone with torch.compile!")
            
    logger.info("LightX2V optimizations applied successfully.")
