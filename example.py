import torch
import os
from tensor_dump import (
    dump_tensor,
    save_tensor_to_bin,
    dump_config,
    load_tensor_from_bin,
    load_tensor_from_txt,
    compare_tensor_dirs
)

# 1. 创建测试张量
x = torch.randn(2, 16).cuda()
os.makedirs("./dump_a", exist_ok=True)
os.makedirs("./dump_b", exist_ok=True)

# 2. Dump txt
dump_tensor(x, "test_x", "./dump_a")
dump_tensor(x, "test_x", "./dump_b")

# 3. Dump bin
save_tensor_to_bin(x, "./dump_a/data.bin")
save_tensor_to_bin(x, "./dump_b/data.bin")

# 4. Dump config
dump_config({
    "x": [list(x.shape), str(x.dtype)]
}, "./dump_a/config.json")

# 5. Load back
t_bin = load_tensor_from_bin("./dump_a/data.bin", shape=(2,16))
t_txt = load_tensor_from_txt("./dump_a/001-test_x_20260419_203306_502968.txt") # 注意替换为真实文件名

print("Tensor 形状:", t_txt.shape)
print("Tensor 类型:", t_txt.dtype)
print("Tensor 完整数据:")
print(t_txt)

# 6. 对比目录（示例）
compare_tensor_dirs("./dump_a", "./dump_b", output_file="result.txt")
