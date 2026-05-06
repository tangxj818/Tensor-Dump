import unittest
import torch
import os
import json
import tempfile
import shutil

from tensor_dump import (
    dump_tensor,
    dump_tensors,
    dump_tensor_to_bin,
    dump_config,
    reset_dump_counter,
    load_tensor_from_txt,
    load_tensor_from_bin,
    compare_tensor_dirs,
    CompareResult,
)


class TestDumpTensor(unittest.TestCase):
    def setUp(self):
        reset_dump_counter()
        self.tmpdir = tempfile.mkdtemp()
        self.tensor = torch.randn(2, 3)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_dump_txt_basic(self):
        path = dump_tensor(self.tensor, "test", self.tmpdir)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith(".txt"))
        with open(path) as f:
            content = f.read()
        self.assertIn("Tensor Name: test", content)
        self.assertIn("Shape: torch.Size([2, 3])", content)
        self.assertIn("Data (first", content)

    def test_dump_txt_scalar(self):
        t = torch.tensor(3.14)
        path = dump_tensor(t, "scalar", self.tmpdir)
        self.assertTrue(os.path.exists(path))
        loaded = load_tensor_from_txt(path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, ())
        self.assertAlmostEqual(loaded.item(), 3.14, places=5)

    def test_dump_txt_no_data(self):
        path = dump_tensor(self.tensor, "test", self.tmpdir, save_data=False)
        self.assertTrue(os.path.exists(path))
        loaded = load_tensor_from_txt(path)
        self.assertIsNone(loaded)

    def test_dump_txt_max_elements(self):
        t = torch.randn(10, 10)
        path = dump_tensor(t, "test", self.tmpdir, max_elements=5)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            content = f.read()
        self.assertIn("Data (first 5 of 100 elements)", content)
        self.assertIn("(100 total elements)", content)

    def test_dump_txt_max_elements_exact(self):
        t = torch.randn(2, 3)
        path = dump_tensor(t, "test", self.tmpdir, max_elements=6)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            content = f.read()
        self.assertIn("Data (first 6 of 6 elements)", content)
        self.assertNotIn("total elements", content)

    def test_dump_txt_device_id_none(self):
        path = dump_tensor(self.tensor, "test", self.tmpdir, device_id=None)
        self.assertTrue(os.path.exists(path))

    def test_dump_txt_device_id_mismatch(self):
        path = dump_tensor(self.tensor, "test", self.tmpdir, device_id=0)
        self.assertEqual(path, "")

    def test_dump_tensors(self):
        tensors = {"a": torch.randn(2, 2), "b": torch.randn(3, 3)}
        dump_tensors(tensors, "grp", self.tmpdir)
        files = sorted(f for f in os.listdir(self.tmpdir) if f.endswith(".txt"))
        self.assertEqual(len(files), 2)
        self.assertIn("grp_a", files[0])
        self.assertIn("grp_b", files[1])

    def test_reset_counter(self):
        p1 = dump_tensor(self.tensor, "a", self.tmpdir)
        p2 = dump_tensor(self.tensor, "b", self.tmpdir)
        reset_dump_counter()
        p3 = dump_tensor(self.tensor, "c", self.tmpdir)
        self.assertTrue(os.path.basename(p1).startswith("001-"))
        self.assertTrue(os.path.basename(p2).startswith("002-"))
        self.assertTrue(os.path.basename(p3).startswith("001-"))

    def test_dump_counter_sequential(self):
        p1 = dump_tensor(self.tensor, "a", self.tmpdir)
        p2 = dump_tensor(self.tensor, "b", self.tmpdir)
        self.assertTrue(os.path.basename(p1).startswith("001-"))
        self.assertTrue(os.path.basename(p2).startswith("002-"))


class TestDumpBin(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_bin_basic(self):
        t = torch.randn(2, 3)
        path = os.path.join(self.tmpdir, "data.bin")
        dump_tensor_to_bin(t, path)
        self.assertTrue(os.path.exists(path))
        loaded = load_tensor_from_bin(path, shape=(2, 3))
        self.assertTrue(torch.allclose(loaded, t, atol=1e-6))

    def test_bin_bf16_cast(self):
        t = torch.randn(2, 3).bfloat16()
        path = os.path.join(self.tmpdir, "data.bin")
        dump_tensor_to_bin(t, path)
        loaded = load_tensor_from_bin(path, shape=(2, 3), dtype=torch.float16)
        t_f16 = t.view(torch.float16).cpu()
        self.assertTrue(torch.allclose(loaded, t_f16, rtol=1e-3, atol=1e-3))

    def test_bin_int(self):
        t = torch.randint(0, 100, (3, 4), dtype=torch.int64)
        path = os.path.join(self.tmpdir, "data.bin")
        dump_tensor_to_bin(t, path)
        loaded = load_tensor_from_bin(path, shape=(3, 4), dtype=torch.int64)
        self.assertTrue(torch.equal(loaded, t.cpu()))

    def test_bin_device_id_mismatch(self):
        path = os.path.join(self.tmpdir, "data.bin")
        dump_tensor_to_bin(torch.randn(2, 3), path, device_id=0)
        self.assertFalse(os.path.exists(path))

    def test_bin_device_id_none(self):
        path = os.path.join(self.tmpdir, "data.bin")
        dump_tensor_to_bin(torch.randn(2, 3), path, device_id=None)
        self.assertTrue(os.path.exists(path))


class TestDumpConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_dump_config(self):
        config = {"shape": [2, 3], "dtype": "float32", "info": {"a": 1}}
        path = os.path.join(self.tmpdir, "config.json")
        dump_config(config, path)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            loaded = json.load(f)
        self.assertEqual(loaded, config)

    def test_dump_config_empty(self):
        path = os.path.join(self.tmpdir, "config.json")
        dump_config({}, path)
        with open(path) as f:
            loaded = json.load(f)
        self.assertEqual(loaded, {})


class TestLoadTxt(unittest.TestCase):
    def setUp(self):
        reset_dump_counter()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_txt_roundtrip(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        p = dump_tensor(t, "test", self.tmpdir)
        loaded = load_tensor_from_txt(p)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, (2, 2))
        self.assertTrue(torch.equal(loaded, t))

    def test_load_txt_1d(self):
        t = torch.tensor([1.5, 2.5, 3.5])
        p = dump_tensor(t, "test", self.tmpdir)
        loaded = load_tensor_from_txt(p)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, (3,))
        self.assertTrue(torch.allclose(loaded, t))

    def test_load_txt_3d(self):
        t = torch.randn(2, 3, 4)
        p = dump_tensor(t, "test", self.tmpdir)
        loaded = load_tensor_from_txt(p)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, (2, 3, 4))
        self.assertTrue(torch.allclose(loaded, t, atol=1e-6))

    def test_load_txt_no_data(self):
        t = torch.randn(2, 3)
        p = dump_tensor(t, "test", self.tmpdir, save_data=False)
        loaded = load_tensor_from_txt(p)
        self.assertIsNone(loaded)

    def test_load_txt_nonexistent(self):
        loaded = load_tensor_from_txt("/nonexistent/file.txt")
        self.assertIsNone(loaded)


class TestCompare(unittest.TestCase):
    def setUp(self):
        reset_dump_counter()
        self.tmpdir1 = tempfile.mkdtemp()
        self.tmpdir2 = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir1)
        shutil.rmtree(self.tmpdir2)

    def test_compare_identical(self):
        t = torch.randn(2, 3)
        dump_tensor(t, "a", self.tmpdir1)
        reset_dump_counter()
        dump_tensor(t, "a", self.tmpdir2)
        results = compare_tensor_dirs(self.tmpdir1, self.tmpdir2)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)

    def test_compare_different(self):
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        dump_tensor(t1, "a", self.tmpdir1)
        reset_dump_counter()
        dump_tensor(t2, "a", self.tmpdir2)
        results = compare_tensor_dirs(self.tmpdir1, self.tmpdir2)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].passed)

    def test_compare_no_common(self):
        dump_tensor(torch.randn(2, 3), "a", self.tmpdir1)
        dump_tensor(torch.randn(2, 3), "b", self.tmpdir2)  # counter is now at 2, no 001 in dir2

    def test_compare_output_file(self):
        t = torch.randn(2, 3)
        dump_tensor(t, "a", self.tmpdir1)
        reset_dump_counter()
        dump_tensor(t, "a", self.tmpdir2)
        out = os.path.join(self.tmpdir1, "result.txt")
        compare_tensor_dirs(self.tmpdir1, self.tmpdir2, output_file=out)
        self.assertTrue(os.path.exists(out))

    def test_compare_rtol_atol(self):
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([1.0, 2.001])
        dump_tensor(t1, "a", self.tmpdir1)
        reset_dump_counter()
        dump_tensor(t2, "a", self.tmpdir2)
        results = compare_tensor_dirs(self.tmpdir1, self.tmpdir2, rtol=1e-3, atol=1e-3)
        self.assertTrue(results[0].passed)

    def test_compare_result_fields(self):
        t = torch.randn(2, 3)
        dump_tensor(t, "a", self.tmpdir1)
        reset_dump_counter()
        dump_tensor(t, "a", self.tmpdir2)
        results = compare_tensor_dirs(self.tmpdir1, self.tmpdir2)
        r = results[0]
        self.assertIsInstance(r, CompareResult)
        self.assertIsInstance(r.sequence_number, int)
        self.assertIsInstance(r.passed, bool)
        self.assertIsInstance(r.max_abs_diff, float)
        self.assertIsInstance(r.shape_match, bool)
        self.assertIsInstance(r.dtype_match, bool)


class TestBinLoad(unittest.TestCase):
    def test_bin_int32(self):
        t = torch.randint(-128, 127, (2, 5), dtype=torch.int32)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "data.bin")
            dump_tensor_to_bin(t, p)
            loaded = load_tensor_from_bin(p, shape=(2, 5), dtype=torch.int32)
            self.assertTrue(torch.equal(loaded, t.cpu()))

    def test_bin_float64(self):
        t = torch.randn(3, 3, dtype=torch.float64)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "data.bin")
            dump_tensor_to_bin(t, p)
            loaded = load_tensor_from_bin(p, shape=(3, 3), dtype=torch.float64)
            self.assertTrue(torch.allclose(loaded, t.cpu()))

    def test_bin_wrong_shape(self):
        t = torch.randn(2, 3)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "data.bin")
            dump_tensor_to_bin(t, p)
            loaded = load_tensor_from_bin(p, shape=(3, 2))
            self.assertEqual(loaded.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
