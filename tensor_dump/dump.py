import os
import json
import sys
from datetime import datetime
from typing import Optional, Union, Dict
import torch

# ========== 全局计数器（txt dump 用）==========
_dump_counter = 0

def _get_next_counter() -> int:
    global _dump_counter
    _dump_counter += 1
    return _dump_counter

def reset_dump_counter():
    global _dump_counter
    _dump_counter = 0

# ========== 1. Dump to TXT（你给的代码）==========
def dump_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    output_dir: str = "/tmp/tensor_dumps",
    save_data: bool = True,
    max_elements: int = sys.maxsize
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    counter = _get_next_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{counter:03d}-{name}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"Tensor Name: {name}\n")
        f.write(f"Sequence Number: {counter}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n\n")

        # 元信息
        f.write(f"Shape: {tensor.shape}\n")
        f.write(f"Dtype: {tensor.dtype}\n")
        f.write(f"Device: {tensor.device}\n")
        f.write(f"Requires Grad: {tensor.requires_grad}\n")
        f.write(f"Is Contiguous: {tensor.is_contiguous()}\n")
        f.write(f"Total Elements: {tensor.numel()}\n\n")

        # 统计
        try:
            tensor_cpu = tensor.detach().cpu()
            f.write(f"Statistics:\n")
            f.write(f"  Min: {tensor_cpu.min().item()}\n")
            f.write(f"  Max: {tensor_cpu.max().item()}\n")
            f.write(f"  Mean: {tensor_cpu.float().mean().item()}\n")
            f.write(f"  Std: {tensor_cpu.float().std().item()}\n")

            has_nan = torch.isnan(tensor_cpu).any().item()
            has_inf = torch.isinf(tensor_cpu).any().item()
            f.write(f"  Has NaN: {has_nan}\n")
            f.write(f"  Has Inf: {has_inf}\n\n")

            if save_data:
                f.write(f"Data (first {max_elements} elements):\n")
                f.write(f"{'-'*60}\n")
                flat = tensor_cpu.flatten()
                n = min(max_elements, flat.numel())
                for i in range(n):
                    f.write(f"  [{i}]: {flat[i].item()}\n")
                if flat.numel() > max_elements:
                    f.write(f"  ... ({flat.numel() - max_elements} more)\n")
        except Exception as e:
            f.write(f"\nError: {e}\n")

    print(f"[DUMP] {counter:03d} {name} -> {filepath}")
    return filepath

def dump_tensors(
    tensors: Dict[str, torch.Tensor],
    prefix: str = "tensor",
    output_dir: str = "/tmp/tensor_dumps",
    **kwargs
):
    for name, tensor in tensors.items():
        full_name = f"{prefix}_{name}"
        dump_tensor(tensor, full_name, output_dir, **kwargs)

# ========== 2. Save to BIN（你给的代码）==========
def save_tensor_to_bin(tensor: torch.Tensor, save_path: str):
    bin_data = tensor.cpu().contiguous().numpy().tobytes()
    with open(save_path, 'wb') as f:
        f.write(bin_data)
    print(f"[BIN] Saved to {save_path}, size: {len(bin_data)} bytes")

# ========== 3. Dump Config JSON（你给的代码）==========
def dump_config(config: dict, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[CONFIG] Saved to {save_path}")