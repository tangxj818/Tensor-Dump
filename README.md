# Tensor Dump Toolkit

完整的张量调试工具：dump、load、compare 三合一

## 功能
- dump tensor to txt（人类可读）
- dump tensor to bin（高精度原始数据）
- dump config to json
- load from bin
- load from txt
- compare two dump directories

## 安装
```bash
cd tensor_dump
pip install .
```

## 使用
```python
from tensor_dump import dump_tensor, save_tensor_to_bin, load_tensor_from_bin, compare_tensor_dirs
```

## 示例

### 1. dump tensor to txt
```python
import torch

tensor = torch.rand(3, 4)
dump_tensor(tensor, 'tensor.txt')
```

### 2. dump tensor to bin
```python
import torch

tensor = torch.rand(3, 4)
save_tensor_to_bin(tensor, 'tensor.bin')
```

### 3. dump config to json
```python
import json

config = {'a': 1, 'b': 2}
with open('config.json', 'w') as f:
    json.dump(config, f)
```

### 4. load from bin
```python
import torch

tensor = load_tensor_from_bin('tensor.bin')
```

### 5. load from txt
```python
import torch

tensor = torch.load('tensor.txt')
```

### 6. compare two dump directories
```python
compare_tensor_dirs('dir1', 'dir2')
```
