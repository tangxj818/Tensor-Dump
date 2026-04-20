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

def load_tensor_from_bin(bin_path: str, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Load tensor from BIN file"""
    np_dtype = TORCH_TO_NUMPY_DTYPE.get(dtype, np.float32)

    with open(bin_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np_dtype)
    
    return torch.tensor(data, dtype=dtype).view(shape)

def load_tensor_from_txt(txt_path: str) -> Optional[torch.Tensor]:
    """
    Load tensor from TXT file
    """
    with open(txt_path, 'r') as f:
        content = f.read()

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