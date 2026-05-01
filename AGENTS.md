# Tensor-Dump

Single-package Python library (`tensor_dump`) for dumping, loading, and comparing PyTorch tensors.

## Setup

```bash
pip install -e .    # editable install
pip install .       # regular install
```

Dependencies: `torch>=1.13`, `numpy>=1.22`.

## Commands

No linter, no formatter, no CI configured. Skip anything that looks for them.

| What | How |
|------|-----|
| Run all tests | `python -m unittest discover -s tests -p "test_*.py" -v` |
| Run a single test class | `python -m unittest tests.test_all.TestDumpTensor -v` |
| Run compare CLI | `python tensor_dump/compare.py <dir1> <dir2> [output_file] [rtol] [atol]` |
| Run example | `python example.py` (requires CUDA — `x.cuda()`) |

## Key caveats

- **Global counter** in `dump_tensor`: sequential counter `_dump_counter` is module-global. Call `reset_dump_counter()` to reset between runs. TXT filenames follow `{counter:03d}-{name}_{timestamp}.txt`.
- **Default dump dir**: `/tmp/tensor_dumps`.
- **bfloat16 → float16**: `dump_tensor_to_bin` casts `bfloat16` to `float16` before writing (no metadata saved). `load_tensor_from_bin` requires explicit `shape` + `dtype`.
- **Version mismatch**: `pyproject.toml` says `0.1.0`, `tensor_dump/__init__.py` says `1.0.0`.

## Package layout

| File | Exports |
|------|---------|
| `tensor_dump/__init__.py` | Public API + `__version__` |
| `tensor_dump/dump.py` | `dump_tensor`, `dump_tensors`, `dump_tensor_to_bin`, `dump_config`, `reset_dump_counter` |
| `tensor_dump/load.py` | `load_tensor_from_txt`, `load_tensor_from_bin` |
| `tensor_dump/compare.py` | `compare_tensor_dirs`, `CompareResult` dataclass, standalone `__main__` CLI |
