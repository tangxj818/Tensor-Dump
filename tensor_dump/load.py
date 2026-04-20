import re
import numpy as np
import torch
from typing import Optional

TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

# ========== 1. 从 BIN 文件加载张量（补齐功能）==========
def load_tensor_from_bin(bin_path: str, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """从 bin 文件加载 tensor"""
    np_dtype = TORCH_TO_NUMPY_DTYPE.get(dtype, np.float32)

    with open(bin_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np_dtype)
    
    return torch.tensor(data, dtype=dtype).view(shape)

# ========== 2. 从 TXT 文件加载张量（补齐功能）==========
def load_tensor_from_txt(txt_path: str) -> Optional[torch.Tensor]:
    """
    从 dump 的 txt 中恢复数据（只恢复保存的前 N 个元素）
    对应 dump_tensor
    """
    with open(txt_path, 'r') as f:
        content = f.read()

    # 正则提取数据
    match = re.search(r"Data \(first \d+ elements\):\n-+\n(.*?)(?:\n\n|\Z)", content, re.DOTALL)
    if not match:
        print(f"[LOAD TXT] No data found in {txt_path}")
        return None

    lines = match.group(1).strip().splitlines()
    values = []
    for line in lines:
        if "]:" in line:
            val = float(line.split("]:")[-1].strip())
            values.append(val)

    tensor = torch.tensor(values)
    print(f"[LOAD TXT] Loaded {txt_path}, elements={len(tensor)}")
    return tensor