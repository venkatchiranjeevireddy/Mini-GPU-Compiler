#!/usr/bin/env python3
"""
Test-suite for mini-ptx.  Run with:  python3 tests/run_tests.py   (or: make test)

For every example kernel it
  1. compiles it twice (optimizer on / off),
  2. assembles the PTX with NVIDIA's real `ptxas` when available (set PTXAS=/path/to/ptxas or put it on PATH),
  3. executes the PTX in a small simulator and compares every output against a Python reference,
and it also checks that bad programs are rejected with sensible error messages.
"""
import math
import os
import random
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ptx_sim as sim  # noqa: E402
from ptx_sim import f32  # noqa: E402

COMPILER = os.path.join(ROOT, "build", "mini-ptx")
EXAMPLES = os.path.join(ROOT, "examples")
PTXAS = os.environ.get("PTXAS") or shutil.which("ptxas")

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print("  PASS  " + name)
    else:
        failed += 1
        print("  FAIL  " + name + (("  -> " + detail) if detail else ""))


def compile_k(path, extra=()):
    return subprocess.run([COMPILER, path, *extra], capture_output=True, text=True)


# ---------------------------------------------------------------- reference computations
def rnd_f(n, lo=-10.0, hi=10.0):
    return [f32(random.uniform(lo, hi)) for _ in range(n)]


def ceil_div(a, b):
    return (a + b - 1) // b


def same_floats(got, exp):
    return all((g == e) or (math.isnan(g) and math.isnan(e)) for g, e in zip(got, exp)) and len(got) == len(exp)


def t_vecadd(K):
    n, bs = 1000, 128
    a, b = rnd_f(n), rnd_f(n)
    mem = sim.Memory()
    pa, pb, pc = mem.alloc_f32(a), mem.alloc_f32(b), mem.alloc_f32([0.0] * n)
    sim.launch(K["vecadd"], [pa, pb, pc, n], mem, (ceil_div(n, bs),), (bs,))
    return same_floats(mem.read_f32(pc), [f32(x + y) for x, y in zip(a, b)])


def t_saxpy(K):
    n, bs, alpha = 777, 64, 2.5
    x, y = rnd_f(n), rnd_f(n)
    mem = sim.Memory()
    px, py = mem.alloc_f32(x), mem.alloc_f32(y)
    sim.launch(K["saxpy"], [n, alpha, px, py], mem, (ceil_div(n, bs),), (bs,))
    exp = [f32(f32(alpha * xi) + yi) for xi, yi in zip(x, y)]
    return same_floats(mem.read_f32(py), exp)


def t_relu(K):
    n, bs = 300, 64
    x = rnd_f(n) + [0.0, -0.0]
    n = len(x)
    mem = sim.Memory()
    pi, po = mem.alloc_f32(x), mem.alloc_f32([9.0] * n)
    sim.launch(K["relu"], [pi, po, n], mem, (ceil_div(n, bs),), (bs,))
    return same_floats(mem.read_f32(po), [v if v > 0 else 0.0 for v in x])


def t_leaky(K):
    n, bs, slope = 200, 32, 0.1
    x = rnd_f(n)
    mem = sim.Memory()
    pi, po = mem.alloc_f32(x), mem.alloc_f32([0.0] * n)
    sim.launch(K["leaky_relu"], [pi, po, n, slope], mem, (ceil_div(n, bs),), (bs,))
    s = f32(slope)
    return same_floats(mem.read_f32(po), [f32(v * s) if v < 0 else v for v in x])


def t_stride(K):
    n, factor = 1000, 1.5
    d = rnd_f(n)
    mem = sim.Memory()
    pd = mem.alloc_f32(d)
    sim.launch(K["scale_stride"], [pd, n, factor], mem, (3,), (64,))  # 192 threads, 1000 elements
    return same_floats(mem.read_f32(pd), [f32(f32(v * factor) + 0.0) for v in d])


def t_matmul(K):
    M, N, Kd = 7, 9, 5
    A, B = rnd_f(M * Kd), rnd_f(Kd * N)
    mem = sim.Memory()
    pA, pB, pC = mem.alloc_f32(A), mem.alloc_f32(B), mem.alloc_f32([0.0] * (M * N))
    sim.launch(K["matmul"], [pA, pB, pC, M, N, Kd], mem, (ceil_div(N, 4), ceil_div(M, 4)), (4, 4))
    exp = []
    for r in range(M):
        for c in range(N):
            acc = 0.0
            for k in range(Kd):
                acc = f32(acc + f32(A[r * Kd + k] * B[k * N + c]))
            exp.append(acc)
    return same_floats(mem.read_f32(pC), exp)


def t_guarded(K):
    n, bs = 100, 128  # 128 threads but only 100 elements: threads 100..127 must NOT load a[i]
    a = rnd_f(n)
    mem = sim.Memory()
    pa, po = mem.alloc_f32(a), mem.alloc_s32([0] * n)
    sim.launch(K["count_positive"], [pa, po, n], mem, (1,), (bs,))
    return mem.read_s32(po) == [1 if v > 0 else 0 for v in a]


def t_collatz(K):
    def ref(x):
        c = 0
        while x != 1 and c < 1000:
            x = x // 2 if x % 2 == 0 else 3 * x + 1
            c += 1
        return c

    n = 150
    starts = list(range(1, n + 1))
    mem = sim.Memory()
    ps, pr = mem.alloc_s32(starts), mem.alloc_s32([-1] * n)
    sim.launch(K["collatz"], [ps, pr, n], mem, (ceil_div(n, 64),), (64,))
    return mem.read_s32(pr) == [ref(x) for x in starts]


def t_features(K):
    def wrap(x):
        x &= 0xFFFFFFFF
        return x - (1 << 32) if x & 0x80000000 else x

    def cdiv(a, b):
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b >= 0) else -q

    def crem(a, b):
        return a - cdiv(a, b) * b

    n = 12
    mem = sim.Memory()
    po, pf = mem.alloc_s32([-1] * (3 * n)), mem.alloc_f32([9.0] * n)
    sim.launch(K["features"], [po, pf, n], mem, (1,), (16,))  # 16 threads, n=12: threads 12..15 return early
    exp_o, exp_f = [0] * (3 * n), []
    for i in range(n):
        v = -i
        if i % 3 == 0 or (5 < i < 8):
            v += 100
        if not (i < 2):
            v *= 2
        w = wrap(2147483647 + i)
        f = f32(float(i) / 2.0)
        exp_o[i] = v
        exp_o[i + n] = wrap(cdiv(w, 7) - crem(w, 7))
        exp_f.append(f32(-f * 3.0))
        exp_o[i + 2 * n] = int(f32(f * 3.0)) - 5
    ok = mem.read_s32(po) == exp_o and same_floats(mem.read_f32(pf), exp_f)
    if not ok:
        print("   got:", mem.read_s32(po), "\n   exp:", exp_o)
    return ok


def t_fill(K):
    n = 37
    mem = sim.Memory()
    po = mem.alloc_s32([0] * n)
    sim.launch(K["fill"], [po, 42, n], mem, (1,), (8,))
    return mem.read_s32(po) == [42] * n


CASES = {
    "features.k": [("features (return, ||, !, wrap, casts)", t_features), ("fill (second kernel in file)", t_fill)],
    "vecadd.k": [("vecadd", t_vecadd)],
    "saxpy.k": [("saxpy", t_saxpy)],
    "relu.k": [("relu", t_relu), ("leaky_relu", t_leaky)],
    "grid_stride.k": [("scale_stride", t_stride)],
    "matmul.k": [("matmul", t_matmul)],
    "guarded.k": [("count_positive (short-circuit)", t_guarded), ("collatz (while loop, int div/rem)", t_collatz)],
}


def run_examples():
    print("\n[1] examples: compile, assemble with ptxas, simulate vs Python reference")
    for fname, cases in CASES.items():
        for mode, flags in (("opt", []), ("no-opt", ["--no-opt"])):
            r = compile_k(os.path.join(EXAMPLES, fname), flags)
            check(f"{fname} [{mode}] compiles", r.returncode == 0, r.stderr)
            if r.returncode != 0:
                continue
            ptx = r.stdout
            if PTXAS:
                tmp = os.path.join(ROOT, "build", f"_t_{fname}_{mode}.ptx")
                with open(tmp, "w") as f:
                    f.write(ptx)
                p = subprocess.run([PTXAS, "-arch=sm_75", tmp, "-o", os.devnull], capture_output=True, text=True)
                check(f"{fname} [{mode}] assembles with ptxas", p.returncode == 0, p.stderr)
                os.remove(tmp)
            kernels = sim.parse_ptx(ptx)
            for label, fn in cases:
                key = label.split()[0]
                random.seed(1234)
                try:
                    ok = fn(kernels)
                    check(f"{fname} [{mode}] {label} matches reference", ok)
                except Exception as e:  # noqa: BLE001
                    check(f"{fname} [{mode}] {label} matches reference", False, repr(e))
    if not PTXAS:
        print("  (ptxas not found - skipped the real-assembler checks; set PTXAS=/path/to/ptxas to enable)")


def run_targets():
    if not PTXAS:
        return
    print("\n[2] other GPU targets assemble with ptxas")
    for sm in ("sm_70", "sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90", "sm_100"):
        r = compile_k(os.path.join(EXAMPLES, "matmul.k"), ["--target", sm])
        tmp = os.path.join(ROOT, "build", f"_t_{sm}.ptx")
        with open(tmp, "w") as f:
            f.write(r.stdout)
        p = subprocess.run([PTXAS, f"-arch={sm}", tmp, "-o", os.devnull], capture_output=True, text=True)
        check(f"matmul.k --target {sm}", r.returncode == 0 and p.returncode == 0, p.stderr)
        os.remove(tmp)


def run_optimizer():
    print("\n[3] optimizer")
    r = compile_k(os.path.join(EXAMPLES, "grid_stride.k"), ["--stats"])
    check("constant folding runs on grid_stride.k", "constant folds" in r.stderr and "0 constant folds" not in r.stderr, r.stderr)
    src = os.path.join(ROOT, "build", "_opt.k")
    with open(src, "w") as f:
        f.write("""kernel k(int* out, int n) {
            int i = threadIdx.x;
            if (1 < 2) { out[i + 0] = i * 1 + 2 * 3; } else { out[i] = 99; }
            while (0 > 1) { out[i] = 7; }
        }""")
    r = compile_k(src, ["--stats", "--dump-ast"])
    check("dead if/else and dead while are removed", "2 dead branches removed" in r.stderr, r.stderr)
    check("x+0 and x*1 are simplified", "2 simplifications" in r.stderr, r.stderr)
    after = r.stderr.split("after optimization")[1]
    check("constant folded result visible in optimized AST", "(+ i 6)" in after and "99" not in after, after)
    os.remove(src)


BAD = [
    ("undeclared variable", "kernel k(int n) { int x = y; }", "undeclared variable 'y'"),
    ("missing semicolon", "kernel k(int n) { int x = 1 }", "expected"),
    ("float to int without cast", "kernel k(int n) { int x = 1.5; }", "explicit cast"),
    ("non-boolean condition", "kernel k(int n) { if (n) { } }", "condition must be a comparison"),
    ("float array index", "kernel k(float* a, float f) { a[f] = 1.0; }", "index must be int"),
    ("redeclaration", "kernel k(int n) { int a = 1; int a = 2; }", "already declared"),
    ("indexing a scalar", "kernel k(int n) { n[0] = 1; }", "not an array"),
    ("pointer used as value", "kernel k(float* a) { float x = a; }", "can only be used with an index"),
    ("unterminated block", "kernel k(int n) { int x = 1;", "unterminated"),
    ("bad character", "kernel k(int n) { int x = 1 @ 2; }", "unexpected character"),
    ("modulo on float", "kernel k(float f) { float x = f % 2.0; }", "only works on int"),
]


def run_errors():
    print("\n[4] compile errors are reported clearly")
    src = os.path.join(ROOT, "build", "_bad.k")
    for name, code, frag in BAD:
        with open(src, "w") as f:
            f.write(code)
        r = compile_k(src)
        ok = r.returncode == 1 and frag in r.stderr and "_bad.k:1:" in r.stderr
        check(name, ok, r.stderr.strip())
    os.remove(src)


def run_host_runner_mock():
    print("\n[5] host/run_gpu.py marshalling, against a fake CUDA driver backed by the simulator")
    import contextlib
    import io

    sys.path.insert(0, os.path.join(ROOT, "host"))
    import fake_cuda
    import run_gpu

    run_gpu.SIZES.update({"vecadd": 1003, "saxpy": 501, "matmul": (9, 7, 5), "collatz": 120})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_gpu.main([], lib=fake_cuda.FakeCuda())
    out = buf.getvalue()
    check("run_gpu.py passes all 4 checks with the fake driver", rc == 0 and out.count("PASS") == 4, out)


def main():
    if not os.path.exists(COMPILER):
        print("build the compiler first:  make")
        return 2
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    print("ptxas:", PTXAS or "not found")
    run_examples()
    run_targets()
    run_optimizer()
    run_errors()
    run_host_runner_mock()
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
