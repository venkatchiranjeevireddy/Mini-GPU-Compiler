# mini-ptx: a small GPU kernel language compiler that emits NVIDIA PTX

A from-scratch compiler in **C++17** (no LLVM, no external libraries). It reads a small C-like kernel language
and generates **PTX** (NVIDIA's GPU virtual ISA) that `ptxas` assembles and the CUDA driver can load and run.

```
kernel source ──► Lexer ──► Parser ──► AST ──► Sema ──► Optimizer ──► PTX codegen ──► .ptx
                  tokens    recursive   │      types,     constant       virtual regs,
                            descent     │      scopes,    folding,       predicates,
                                        │      int→float  dead-branch    ld/st.global,
                                        │      casts      elimination    short-circuit &&/||
                                     --dump-ast (before / after optimization)
```

## Quick start

```bash
make                                   # builds build/mini-ptx (g++ with -Wall -Wextra -Werror)
build/mini-ptx examples/vecadd.k       # PTX to stdout
build/mini-ptx examples/matmul.k -o matmul.ptx --target sm_80
build/mini-ptx examples/grid_stride.k --dump-ast --stats -o /dev/null
make test                              # full test-suite (see "How it is verified")
```

Options: `-o file`, `--target sm_70|sm_75|sm_80|sm_86|sm_89|sm_90|sm_100`, `--no-opt`, `--dump-ast`, `--stats`.

## The language

```c
kernel saxpy(int n, float alpha, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}
```

| Feature | Details |
|---|---|
| Types | `int` (32-bit), `float` (32-bit); kernel parameters may also be `int*` / `float*` (global memory) |
| Built-ins | `threadIdx`, `blockIdx`, `blockDim`, `gridDim` with `.x .y .z` |
| Statements | declarations, `=`, `+= -= *= /=`, `x++ x--`, `a[i] = ...`, `if / else`, `while`, `for`, `return` |
| Expressions | `+ - * / %`, comparisons, `&& ||` (short-circuit), `!`, unary `-`, casts `(int)` `(float)` |
| Conversions | `int → float` is implicit; `float → int` needs an explicit `(int)` cast (compile error otherwise) |
| Conditions | must be comparisons (`i < n`), like a strict subset of C |
| Multiple kernels | any number of `kernel` definitions per file |

Not supported (by design, to keep the compiler small and fully tested): shared memory, `__syncthreads`, atomics,
function calls / math intrinsics, structs, 64-bit types, `break`/`continue`, pointer arithmetic.

## What the compiler does

* **Lexer**: hand-written, tracks line/column, handles `//` and `/* */` comments, `1.5f`, `1e-3`, `.5`.
* **Parser**: recursive descent with one function per precedence level. `for` is desugared to
  `{ init; while (cond) { body; step; } }`; `x += e` becomes `x = x + e`.
* **Semantic analysis**: block scopes, symbol table, full type checking, implicit `int→float` conversions made
  explicit as `Cast` nodes. Errors carry `file:line:col` plus the source line and a caret.
* **Optimizer** (all expressions are side-effect free, so every rewrite is safe):
  constant folding with exact C semantics (32-bit wrap-around, truncating `/` and `%`, refuses to fold
  division by zero or `INT_MIN / -1`, never bakes `inf`/`NaN` into the output), algebraic identities
  (`x+0`, `x*1`, `x*0` for ints; `x*1.0`, `x/1.0` for floats, only the ones that are exact for every float),
  `true && x → x`, and dead `if`/`while` branch elimination.
* **Code generator**: virtual registers (`%r` int, `%f` float, `%rd` 64-bit address, `%p` predicate),
  parameters loaded with `ld.param` + `cvta.to.global`, element addressing with `mul.wide.s32` + `add.s64`,
  `ld.global` / `st.global`, comparisons as `setp` predicates, control flow as predicated `bra`.
  * `&&` / `||` are compiled to **real short-circuit branches**, so `if (i < n && a[i] > 0.0)` never reads
    `a[i]` out of bounds.
  * Float arithmetic uses `add.rn.f32`, `mul.rn.f32`, ... (IEEE round-to-nearest, no fused multiply-add), so GPU
    results match a plain CPU computation bit for bit.
  * Float `!=` uses `setp.neu` so that `NaN != NaN` is true, matching C.
  * The `.version` directive is chosen per `--target` (ptxas rejects a `.target` newer than the `.version`).

### Example: input and output

`examples/vecadd.k` →

```ptx
.visible .entry vecadd(.param .u64 vecadd_param_0, ... .param .u32 vecadd_param_3)
{
    ...
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mul.lo.s32 %r5, %r3, %r4;
    mov.u32 %r6, %tid.x;
    add.s32 %r7, %r5, %r6;
    mov.s32 %r2, %r7;
    setp.lt.s32 %p1, %r2, %r1;
    @!%p1 bra $L_endif2;
    ...
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd8];
    add.rn.f32 %f3, %f1, %f2;
    st.global.f32 [%rd4], %f3;
$L_endif2:
    ret;
}
```

Every example's generated PTX is in `examples/ptx/`.

## How it is verified

`make test` (or `python3 tests/run_tests.py`) runs 70+ checks:

1. **Real assembler**: every example (optimizer on and off) is assembled by NVIDIA's `ptxas`
   (`pip install nvidia-cuda-nvcc-cu12` provides one, or use the CUDA toolkit; set `PTXAS=/path/to/ptxas`).
   The same kernel is assembled for sm_70, 75, 80, 86, 87, 89, 90 and 100.
2. **Behaviour**: `tests/ptx_sim.py` is a small PTX interpreter. Each kernel runs over a full launch grid with
   bounds-checked global memory and float32 rounding after every operation, and the output is compared with a
   Python reference. Kernels: vector add, SAXPY, ReLU / leaky ReLU, grid-stride loop, naive matrix multiply,
   short-circuit guard, Collatz (while loop, int `/` and `%`), plus a feature tour (early `return`, `||`, `!`,
   int wrap-around, casts).
3. **Optimizer** correctness: optimized and unoptimized builds must produce identical results.
4. **Diagnostics**: 11 bad programs must be rejected with the right message and `file:line:col`.
5. **Host runner**: `host/run_gpu.py` is exercised against a fake CUDA driver backed by the simulator.

The compiler is also clean under AddressSanitizer + UBSan.

## Running on a real GPU (Google Colab, free T4)

Runtime → Change runtime type → T4 GPU, then:

```bash
!git clone https://github.com/<you>/mini-ptx-compiler && cd mini-ptx-compiler && make
!cd mini-ptx-compiler && python3 host/run_gpu.py
```

`host/run_gpu.py` calls the CUDA driver API through `ctypes` (`cuModuleLoadDataEx`, `cuLaunchKernel`, ...),
runs vector add, SAXPY, matrix multiply and Collatz on the GPU, and compares with CPU results.

You can also compare against NVIDIA's compiler: write the same kernel in CUDA C++ and run `nvcc -ptx` to see
how `nvcc` lowers it.

## Project layout

```
src/lexer.hpp      tokens           src/sema.hpp       scopes + type checking
src/parser.hpp     AST              src/optimizer.hpp  folding + dead-branch elimination
src/ast.hpp        AST + printer    src/codegen.hpp    PTX emission
src/main.cpp       CLI              host/run_gpu.py    run PTX on a GPU via the driver API
tests/             simulator, fake driver, test-suite   examples/  kernels (+ examples/ptx output)
```

## Known limitations / ideas

* Single-pass tree-walking code generator with no register allocation of its own (ptxas allocates registers),
  so the PTX is straightforward rather than minimal. A next step is lowering to SSA IR and adding CSE and
  loop-invariant code motion.
* No shared memory / barriers / atomics yet, so reductions are not expressible.
* Local variables without an initializer are zero-initialised.
# Mini-GPU-Compiler
