#!/usr/bin/env python3
"""
run_gpu.py - run the PTX produced by mini-ptx on a real NVIDIA GPU and check it against a CPU reference.

It talks to the CUDA *driver* API (libcuda) through ctypes, the same calls a C++ host program would make:
cuModuleLoadDataEx -> cuModuleGetFunction -> cuMemAlloc -> cuMemcpyHtoD -> cuLaunchKernel -> cuMemcpyDtoH.
Nothing needs to be installed besides the NVIDIA driver, so it works as-is on Google Colab (GPU runtime).

    make                      # build the compiler first
    python3 host/run_gpu.py

Options:  --target sm_XX   (default sm_75; PTX is JIT-compiled by the driver so newer GPUs work too)
"""
import array
import ctypes
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from ptx_sim import f32  # noqa: E402  (float32 rounding helper for the CPU reference)

COMPILER = os.path.join(ROOT, "build", "mini-ptx")
c_void_p, c_int, c_uint, c_uint64, c_size_t, c_float, c_char_p = (
    ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_float, ctypes.c_char_p)
P = ctypes.POINTER


def load_driver():
    for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def declare_signatures(lib):
    sigs = {
        "cuInit": [c_uint],
        "cuDeviceGet": [P(c_int), c_int],
        "cuDeviceGetName": [c_char_p, c_int, c_int],
        "cuDevicePrimaryCtxRetain": [P(c_void_p), c_int],
        "cuCtxSetCurrent": [c_void_p],
        "cuModuleLoadDataEx": [P(c_void_p), c_char_p, c_uint, P(c_int), P(c_void_p)],
        "cuModuleGetFunction": [P(c_void_p), c_void_p, c_char_p],
        "cuMemAlloc_v2": [P(c_uint64), c_size_t],
        "cuMemFree_v2": [c_uint64],
        "cuMemcpyHtoD_v2": [c_uint64, c_void_p, c_size_t],
        "cuMemcpyDtoH_v2": [c_void_p, c_uint64, c_size_t],
        "cuLaunchKernel": [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p, P(c_void_p), c_void_p],
        "cuCtxSynchronize": [],
    }
    for fname, argtypes in sigs.items():
        getattr(lib, fname).argtypes = argtypes
        getattr(lib, fname).restype = c_int


class Gpu:
    """Thin wrapper over the driver API calls we need."""

    def __init__(self, lib):
        self.lib = lib
        declare_signatures(lib)
        self.ok(lib.cuInit(0), "cuInit")
        dev = c_int()
        self.ok(lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        name = ctypes.create_string_buffer(256)
        lib.cuDeviceGetName(name, 256, dev.value)
        self.device_name = name.value.decode(errors="replace")
        ctx = c_void_p()
        self.ok(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value), "cuDevicePrimaryCtxRetain")
        self.ok(lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    @staticmethod
    def ok(code, what):
        if code != 0:
            raise RuntimeError(f"CUDA driver call {what} failed with error code {code}")

    def load_module(self, ptx_text):
        log = ctypes.create_string_buffer(8192)
        opts = (c_int * 2)(5, 6)  # CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        vals = (c_void_p * 2)(ctypes.addressof(log), len(log))
        mod = c_void_p()
        code = self.lib.cuModuleLoadDataEx(ctypes.byref(mod), ptx_text.encode() + b"\0", 2, opts, vals)
        if code != 0:
            raise RuntimeError(f"driver could not load the PTX (error {code}):\n{log.value.decode(errors='replace')}")
        return mod

    def get_function(self, mod, name):
        fn = c_void_p()
        self.ok(self.lib.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()), "cuModuleGetFunction " + name)
        return fn

    def upload(self, arr):
        """Copy an array.array to the GPU, return the device pointer."""
        nbytes = len(arr) * arr.itemsize
        d = c_uint64()
        self.ok(self.lib.cuMemAlloc_v2(ctypes.byref(d), nbytes), "cuMemAlloc")
        self.ok(self.lib.cuMemcpyHtoD_v2(d.value, c_void_p(arr.buffer_info()[0]), nbytes), "cuMemcpyHtoD")
        return d.value

    def download(self, dptr, typecode, count):
        out = array.array(typecode, bytes(count * array.array(typecode).itemsize))
        self.ok(self.lib.cuMemcpyDtoH_v2(c_void_p(out.buffer_info()[0]), dptr, len(out) * out.itemsize), "cuMemcpyDtoH")
        return out

    def free(self, dptr):
        self.lib.cuMemFree_v2(dptr)

    def launch(self, fn, grid, block, args):
        """args: list of ('ptr', dptr) | ('i32', int) | ('f32', float), in kernel parameter order."""
        holders = []
        for kind, v in args:
            holders.append(c_uint64(v) if kind == "ptr" else c_int(v) if kind == "i32" else c_float(v))
        params = (c_void_p * len(holders))(*[ctypes.cast(ctypes.pointer(h), c_void_p) for h in holders])
        g = (list(grid) + [1, 1])[:3]
        b = (list(block) + [1, 1])[:3]
        self.ok(self.lib.cuLaunchKernel(fn, g[0], g[1], g[2], b[0], b[1], b[2], 0, None, params, None), "cuLaunchKernel")
        self.ok(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")


# ---------------------------------------------------------------- tests (CPU reference vs GPU result)
def ceil_div(a, b):
    return (a + b - 1) // b


def max_err(got, exp):
    worst = 0.0
    for g, e in zip(got, exp):
        worst = max(worst, abs(g - e) / max(1.0, abs(e)))
    return worst


def compile_example(name, target):
    src = os.path.join(ROOT, "examples", name)
    r = subprocess.run([COMPILER, src, "--target", target], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout


# Problem sizes (the mock-driver test in tests/run_tests.py shrinks these so the Python simulator stays fast).
SIZES = {"vecadd": 100_003, "saxpy": 50_001, "matmul": (67, 53, 41), "collatz": 5000}


def test_vecadd(gpu, ptx):
    import random
    random.seed(1)
    n, bs = SIZES["vecadd"], 256
    a = array.array("f", [random.uniform(-5, 5) for _ in range(n)])
    b = array.array("f", [random.uniform(-5, 5) for _ in range(n)])
    da, db, dc = gpu.upload(a), gpu.upload(b), gpu.upload(array.array("f", [0.0] * n))
    gpu.launch(gpu.get_function(gpu.load_module(ptx), "vecadd"), (ceil_div(n, bs),), (bs,),
               [("ptr", da), ("ptr", db), ("ptr", dc), ("i32", n)])
    got = gpu.download(dc, "f", n)
    err = max_err(got, [f32(x + y) for x, y in zip(a, b)])
    for d in (da, db, dc):
        gpu.free(d)
    return err, f"vecadd n={n}"


def test_saxpy(gpu, ptx):
    import random
    random.seed(2)
    n, bs, alpha = SIZES["saxpy"], 128, 2.5
    x = array.array("f", [random.uniform(-5, 5) for _ in range(n)])
    y = array.array("f", [random.uniform(-5, 5) for _ in range(n)])
    dx, dy = gpu.upload(x), gpu.upload(y)
    gpu.launch(gpu.get_function(gpu.load_module(ptx), "saxpy"), (ceil_div(n, bs),), (bs,),
               [("i32", n), ("f32", alpha), ("ptr", dx), ("ptr", dy)])
    got = gpu.download(dy, "f", n)
    err = max_err(got, [f32(f32(alpha * xi) + yi) for xi, yi in zip(x, y)])
    gpu.free(dx)
    gpu.free(dy)
    return err, f"saxpy n={n}"


def test_matmul(gpu, ptx):
    import random
    random.seed(3)
    M, N, K = SIZES["matmul"]
    A = array.array("f", [random.uniform(-1, 1) for _ in range(M * K)])
    B = array.array("f", [random.uniform(-1, 1) for _ in range(K * N)])
    dA, dB, dC = gpu.upload(A), gpu.upload(B), gpu.upload(array.array("f", [0.0] * (M * N)))
    gpu.launch(gpu.get_function(gpu.load_module(ptx), "matmul"), (ceil_div(N, 16), ceil_div(M, 16)), (16, 16),
               [("ptr", dA), ("ptr", dB), ("ptr", dC), ("i32", M), ("i32", N), ("i32", K)])
    got = gpu.download(dC, "f", M * N)
    exp = []
    for r in range(M):
        for c in range(N):
            acc = 0.0
            for k in range(K):
                acc = f32(acc + f32(A[r * K + k] * B[k * N + c]))
            exp.append(acc)
    for d in (dA, dB, dC):
        gpu.free(d)
    return max_err(got, exp), f"matmul {M}x{K} * {K}x{N}"


def test_collatz(gpu, ptx):
    def ref(x):
        c = 0
        while x != 1 and c < 1000:
            x = x // 2 if x % 2 == 0 else 3 * x + 1
            c += 1
        return c

    n, bs = SIZES["collatz"], 128
    starts = array.array("i", range(1, n + 1))
    ds, dr = gpu.upload(starts), gpu.upload(array.array("i", [-1] * n))
    gpu.launch(gpu.get_function(gpu.load_module(ptx), "collatz"), (ceil_div(n, bs),), (bs,),
               [("ptr", ds), ("ptr", dr), ("i32", n)])
    got = gpu.download(dr, "i", n)
    bad = sum(1 for g, s in zip(got, starts) if g != ref(s))
    gpu.free(ds)
    gpu.free(dr)
    return float(bad), f"collatz n={n}"


TESTS = [("vecadd.k", test_vecadd), ("saxpy.k", test_saxpy), ("matmul.k", test_matmul), ("guarded.k", test_collatz)]


def main(argv=None, lib=None):
    argv = sys.argv[1:] if argv is None else argv
    target = "sm_75"
    if "--target" in argv:
        target = argv[argv.index("--target") + 1]
    if not os.path.exists(COMPILER):
        print("compiler not built yet - run `make` first")
        return 2
    lib = lib or load_driver()
    if lib is None:
        print("Could not find the NVIDIA driver library (libcuda).\n"
              "Run this on a machine with an NVIDIA GPU - e.g. Google Colab with Runtime > Change runtime type > T4 GPU.")
        return 2
    gpu = Gpu(lib)
    print("GPU:", gpu.device_name)
    failures = 0
    for example, fn in TESTS:
        try:
            err, label = fn(gpu, compile_example(example, target))
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {example}: {e}")
            failures += 1
            continue
        ok = err <= 1e-6
        failures += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label}  (max error vs CPU: {err:.2e})")
    print("all GPU checks passed" if failures == 0 else f"{failures} GPU check(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
