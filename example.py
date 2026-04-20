import torch
import os
from tensor_dump import (
    dump_tensor,
    dump_tensor_to_bin,
    dump_config,
    load_tensor_from_bin,
    load_tensor_from_txt,
    compare_tensor_dirs
)

# 1. Create test tensors
x = torch.randn(2, 16).cuda()
os.makedirs("./dump_a", exist_ok=True)
os.makedirs("./dump_b", exist_ok=True)

# 2. Dump txt
dump_tensor(x, "test_x", "./dump_a")
dump_tensor(x, "test_x", "./dump_b")

# 3. Dump bin
dump_tensor_to_bin(x, "./dump_a/data.bin")
dump_tensor_to_bin(x, "./dump_b/data.bin")

# 4. Dump config
dump_config({
    "x": [list(x.shape), str(x.dtype)]
}, "./dump_a/config.json")

# 5. Load back
t_bin = load_tensor_from_bin("./dump_a/data.bin", shape=(2,16))
t_txt = load_tensor_from_txt("./dump_a/001-test_x_20260419_203306_502968.txt") # Note: replace with actual filename

print("Tensor 形状:", t_txt.shape)
print("Tensor 类型:", t_txt.dtype)
print("Tensor 完整数据:")
print(t_txt)

# 6. Compare directories
compare_tensor_dirs("./dump_a", "./dump_b", output_file="result.txt")
