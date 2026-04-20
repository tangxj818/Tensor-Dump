# Tensor Dump Toolkit

Complete Tensor Dump Tool: dump, load, compare all-in-one

## Function
- dump tensor to txt
- dump tensor to bin
- dump config to json
- load from bin
- load from txt
- compare two dump directories

## Install
```bash
cd Tensor-Dump
pip install .
```

## Usage
```python
import tensor_dump
from tensor_dump import (
    dump_tensor,
    dump_tensor_to_bin,
    dump_config,
    load_tensor_from_bin,
    load_tensor_from_txt,
    compare_tensor_dirs
)
```
## Example
```bash
python example.py
```

## API Reference
### dump_tensor
Export a PyTorch tensor to a readable TXT file, including shape, type, statistical information, and data content.
```python
def dump_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    output_dir: str = "/tmp/tensor_dumps",
    save_data: bool = True,
    max_elements: int = sys.maxsize
) -> str:
```
- name: Tensor name (for file name)
- save_data: Whether to save specific values
- max_elements: How many elements can be stored at most

### dump_tensor_to_bin
Save the tensor in its original binary format (bin), lossless and high-precision, suitable for C/CUDA alignment.
```python
def dump_tensor_to_bin(tensor: torch.Tensor, save_path: str):
```


### dump_config
Export configuration information (shape, dtype, parameters, sequence information) as JSON.
```python
def dump_config(config: Dict, save_path: str):
```

### load_tensor_from_bin
To load a tensor from a bin file, you must provide the shape and dtype to correctly restore it.
```python
def load_tensor_from_bin(bin_path: str, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
```

### load_tensor_from_txt
Extract and recover tensor data from TXT files generated from a dump.
```python
def load_tensor_from_txt(txt_path: str) -> Optional[torch.Tensor]:
```

### compare_tensor_dirs
Automatically compare TXT dump files in two directories, match them by sequence number, and output a comparison report.
```python
def compare_tensor_dirs(
    dir1: str,
    dir2: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    output_file: Optional[str] = None
) -> list[CompareResult]:
```
- dir1: first dump directory
- dir2: second dump directory
- rtol: relative error tolerance
- atol: absolute error tolerance