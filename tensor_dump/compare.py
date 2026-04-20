"""Utility for comparing tensor dumps from different directories."""
import os
import re
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import torch


@dataclass
class TensorInfo:
    """Store tensor information read from a file"""
    name: str
    sequence_number: int
    shape: tuple
    dtype: str
    data: torch.Tensor
    filepath: str


@dataclass
class CompareResult:
    """tensor comparison results"""
    sequence_number: int
    name1: str
    name2: str
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    mean_rel_diff: float
    shape_match: bool
    dtype_match: bool
    has_nan: bool
    has_inf: bool
    message: str


def parse_tensor_file(filepath: str) -> Optional[TensorInfo]:
    """
    Parse a tensor dump file and extract key information.
    Parameters:
    filepath: Path to the tensor dump file
    Returns:
    TensorInfo object, or None if parsing fails
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        name_match = re.search(r'Tensor Name: (.+)', content)
        name = name_match.group(1).strip() if name_match else "unknown"

        seq_match = re.search(r'Sequence Number: (\d+)', content)
        sequence_number = int(seq_match.group(1)) if seq_match else -1

        shape_match = re.search(r'Shape: torch\.Size\(\[([^\]]*)\]\)', content)
        if shape_match:
            shape_str = shape_match.group(1)
            shape = tuple(map(int, shape_str.split(', '))) if shape_str else ()
        else:
            shape = ()

        dtype_match = re.search(r'Dtype: (.+)', content)
        dtype = dtype_match.group(1).strip() if dtype_match else "unknown"

        data_section = re.search(r'Data \(first \d+ elements\):\n-+\n(.*?)(?:\n\n|\Z)', content, re.DOTALL)
        if data_section:
            data_lines = data_section.group(1).strip().split('\n')
            values = []
            for line in data_lines:
                if line.strip().startswith('[') and ']:' in line:
                    value_str = line.split(']:')[1].strip()
                    try:
                        values.append(float(value_str))
                    except ValueError:
                        pass  
            
            if values:
                data = torch.tensor(values)
            else:
                data = torch.tensor([])
        else:
            data = torch.tensor([])
        
        return TensorInfo(
            name=name,
            sequence_number=sequence_number,
            shape=shape,
            dtype=dtype,
            data=data,
            filepath=filepath
        )
    
    except Exception as e:
        print(f"警告: 无法解析文件 {filepath}: {e}")
        return None


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> Tuple[bool, dict]:
    """
    Compare the numerical differences between two tensors.

    Parameters:
    tensor1: The first tensor
    tensor2: The second tensor
    rtol: Relative error tolerance
    atol: Absolute error tolerance

    Returns:
    (whether it passes, detailed statistics)
    """
    if tensor1.shape != tensor2.shape:
        return False, {
            'max_abs_diff': float('inf'),
            'max_rel_diff': float('inf'),
            'mean_abs_diff': float('inf'),
            'mean_rel_diff': float('inf'),
            'has_nan': False,
            'has_inf': False,
        }

    tensor1 = tensor1.float()
    tensor2 = tensor2.float()

    has_nan = torch.isnan(tensor1).any() or torch.isnan(tensor2).any()
    has_inf = torch.isinf(tensor1).any() or torch.isinf(tensor2).any()

    abs_diff = torch.abs(tensor1 - tensor2)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    denominator = torch.maximum(torch.abs(tensor1), torch.abs(tensor2))
    denominator = torch.maximum(denominator, torch.tensor(atol))
    rel_diff = abs_diff / denominator
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    passed = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol) and not has_nan and not has_inf
    
    stats = {
        'max_abs_diff': max_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_abs_diff': mean_abs_diff,
        'mean_rel_diff': mean_rel_diff,
        'has_nan': has_nan.item() if isinstance(has_nan, torch.Tensor) else has_nan,
        'has_inf': has_inf.item() if isinstance(has_inf, torch.Tensor) else has_inf,
    }
    
    return passed, stats


def get_sequence_number_from_filename(filename: str) -> int:
    """Extract the serial number from the file name"""
    match = re.match(r'(\d+)-', filename)
    return int(match.group(1)) if match else -1


def compare_tensor_dirs(
    dir1: str,
    dir2: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    output_file: Optional[str] = None
) -> list[CompareResult]:
    """
    Compare tensor files in two directories.

    Parameters:
    dir1: The first directory (e.g., output from NVIDIA GPU)
    dir2: The second directory (e.g., output from AMD GPU)
    rtol: Relative error tolerance
    atol: Absolute error tolerance
    output_file: Optional, the file path to save the comparison results

    Returns:
    A list of CompareResult
    """
    dir1_files = {}
    dir2_files = {}
    
    for filename in os.listdir(dir1):
        if filename.endswith('.txt'):
            seq_num = get_sequence_number_from_filename(filename)
            if seq_num >= 0:
                dir1_files[seq_num] = os.path.join(dir1, filename)
    
    for filename in os.listdir(dir2):
        if filename.endswith('.txt'):
            seq_num = get_sequence_number_from_filename(filename)
            if seq_num >= 0:
                dir2_files[seq_num] = os.path.join(dir2, filename)

    common_seqs = sorted(set(dir1_files.keys()) & set(dir2_files.keys()))
    
    if not common_seqs:
        print(f"警告: 在两个目录中没有找到相同序号的文件")
        print(f"  {dir1}: {sorted(dir1_files.keys())}")
        print(f"  {dir2}: {sorted(dir2_files.keys())}")
        return []
    
    print(f"找到 {len(common_seqs)} 对相同序号的文件")
    print(f"序号范围: {min(common_seqs)} - {max(common_seqs)}")
    print(f"{'='*80}")
    
    results = []
    
    for seq_num in common_seqs:
        info1 = parse_tensor_file(dir1_files[seq_num])
        info2 = parse_tensor_file(dir2_files[seq_num])
        
        if info1 is None or info2 is None:
            results.append(CompareResult(
                sequence_number=seq_num,
                name1=info1.name if info1 else "unknown",
                name2=info2.name if info2 else "unknown",
                passed=False,
                max_abs_diff=float('inf'),
                max_rel_diff=float('inf'),
                mean_abs_diff=float('inf'),
                mean_rel_diff=float('inf'),
                shape_match=False,
                dtype_match=False,
                has_nan=False,
                has_inf=False,
                message="文件解析失败"
            ))
            continue

        shape_match = info1.shape == info2.shape
        dtype_match = info1.dtype == info2.dtype

        if len(info1.data) == 0 or len(info2.data) == 0:
            results.append(CompareResult(
                sequence_number=seq_num,
                name1=info1.name,
                name2=info2.name,
                passed=True,  
                max_abs_diff=0.0,
                max_rel_diff=0.0,
                mean_abs_diff=0.0,
                mean_rel_diff=0.0,
                shape_match=shape_match,
                dtype_match=dtype_match,
                has_nan=False,
                has_inf=False,
                message="无数据可对比（仅元信息）"
            ))
            continue

        passed, stats = compare_tensors(info1.data, info2.data, rtol=rtol, atol=atol)

        if passed:
            message = "✓ PASS"
        else:
            reasons = []
            if not shape_match:
                reasons.append(f"shape不匹配: {info1.shape} vs {info2.shape}")
            if stats['has_nan']:
                reasons.append("包含NaN")
            if stats['has_inf']:
                reasons.append("包含Inf")
            if stats['max_abs_diff'] > atol or stats['max_rel_diff'] > rtol:
                reasons.append(f"误差超限 (max_abs={stats['max_abs_diff']:.2e}, max_rel={stats['max_rel_diff']:.2e})")
            message = "✗ FAIL: " + "; ".join(reasons)
        
        result = CompareResult(
            sequence_number=seq_num,
            name1=info1.name,
            name2=info2.name,
            passed=passed,
            max_abs_diff=stats['max_abs_diff'],
            max_rel_diff=stats['max_rel_diff'],
            mean_abs_diff=stats['mean_abs_diff'],
            mean_rel_diff=stats['mean_rel_diff'],
            shape_match=shape_match,
            dtype_match=dtype_match,
            has_nan=stats['has_nan'],
            has_inf=stats['has_inf'],
            message=message
        )
        
        results.append(result)

    print_compare_results(results)

    if output_file:
        save_compare_results(results, output_file)
        print(f"\n对比结果已保存到: {output_file}")
    
    return results


def print_compare_results(results: list[CompareResult]):
    """打印对比结果"""
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    
    print(f"\n{'='*80}")
    print(f"对比结果汇总:")
    print(f"  总计: {len(results)} 对")
    print(f"  通过: {passed_count} 对")
    print(f"  失败: {failed_count} 对")
    print(f"{'='*80}\n")
    
    if failed_count > 0:
        print("失败的对比（误差较大）:\n")
        for result in results:
            if not result.passed:
                print(f"[{result.sequence_number:03d}] {result.name1} vs {result.name2}")
                print(f"  {result.message}")
                print(f"  最大绝对误差: {result.max_abs_diff:.6e}")
                print(f"  最大相对误差: {result.max_rel_diff:.6e}")
                print(f"  平均绝对误差: {result.mean_abs_diff:.6e}")
                print(f"  平均相对误差: {result.mean_rel_diff:.6e}")
                print()

    print("所有对比结果:")
    print(f"{'序号':<6} {'名称1':<25} {'名称2':<25} {'结果':<10} {'最大绝对误差':<15} {'最大相对误差':<15}")
    print("-" * 100)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.sequence_number:03d}    {result.name1:<25} {result.name2:<25} {status:<10} "
              f"{result.max_abs_diff:<15.6e} {result.max_rel_diff:<15.6e}")


def save_compare_results(results: list[CompareResult], output_file: str):
    """保存对比结果到文件"""
    with open(output_file, 'w') as f:
        f.write("Tensor 对比结果\n")
        f.write("=" * 80 + "\n\n")
        
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        
        f.write(f"汇总:\n")
        f.write(f"  总计: {len(results)} 对\n")
        f.write(f"  通过: {passed_count} 对\n")
        f.write(f"  失败: {failed_count} 对\n\n")
        f.write("=" * 80 + "\n\n")
        
        if failed_count > 0:
            f.write("失败的对比:\n\n")
            for result in results:
                if not result.passed:
                    f.write(f"[{result.sequence_number:03d}] {result.name1} vs {result.name2}\n")
                    f.write(f"  {result.message}\n")
                    f.write(f"  最大绝对误差: {result.max_abs_diff:.6e}\n")
                    f.write(f"  最大相对误差: {result.max_rel_diff:.6e}\n")
                    f.write(f"  平均绝对误差: {result.mean_abs_diff:.6e}\n")
                    f.write(f"  平均相对误差: {result.mean_rel_diff:.6e}\n")
                    f.write(f"  Shape匹配: {result.shape_match}\n")
                    f.write(f"  Dtype匹配: {result.dtype_match}\n")
                    f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("详细结果:\n\n")
        f.write(f"{'序号':<6} {'名称1':<25} {'名称2':<25} {'结果':<10} {'最大绝对误差':<15} {'最大相对误差':<15}\n")
        f.write("-" * 100 + "\n")
        
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            f.write(f"{result.sequence_number:03d}    {result.name1:<25} {result.name2:<25} {status:<10} "
                   f"{result.max_abs_diff:<15.6e} {result.max_rel_diff:<15.6e}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("使用方法: python tensor_compare.py <dir1> <dir2> [output_file] [rtol] [atol]")
        print("示例: python tensor_compare.py /tmp/nvidia_dumps /tmp/amd_dumps result.txt 1e-5 1e-8")
        sys.exit(1)
    
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    rtol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    atol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-8
    
    compare_tensor_dirs(dir1, dir2, rtol=rtol, atol=atol, output_file=output_file)
