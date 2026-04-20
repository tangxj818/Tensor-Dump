import re
import numpy as np
import torch
from typing import Optional

# ========== 1. 从 BIN 文件加载张量（补齐功能）==========
def load_tensor_from_bin(
    bin_path: str,
    shape,
    dtype=torch.float32
) -> torch.Tensor:
    """
    从 bin 加载原始张量（必须知道 shape + dtype）
    对应 save_tensor_to_bin
    """
    with open(bin_path, 'rb') as f:
        buffer = f.read()

    # numpy 解析
    np_dtype = dtype.numpy() if isinstance(dtype, torch.dtype) else dtype
    arr = np.frombuffer(buffer, dtype=np_dtype)
    tensor = torch.from_numpy(arr).reshape(shape)
    print(f"[LOAD BIN] Loaded {bin_path}, shape={tensor.shape}, dtype={tensor.dtype}")
    return tensor

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