"""
ptx_sim.py - a tiny PTX interpreter that understands exactly the instruction subset mini-ptx emits.

It lets the test-suite check *behaviour* (not just that the text assembles) on any machine, without a GPU:
every thread of the launch grid is executed one after another, global memory is bounds-checked, and
float32 rounding is applied after every float operation, so results can be compared bit-for-bit
against a Python reference.
"""
import math
import re
import struct


def f32(x):
    """Round a Python float to IEEE float32."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def wrap32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


class MemoryFault(Exception):
    pass


class Memory:
    """Global memory made of separate allocations; any access outside an allocation raises MemoryFault."""

    def __init__(self):
        self.allocs = []  # (base, size, bytearray)
        self.next_base = 0x10000

    def alloc_f32(self, values):
        data = bytearray(struct.pack("<%df" % len(values), *values))
        return self._add(data)

    def alloc_s32(self, values):
        data = bytearray(struct.pack("<%di" % len(values), *values))
        return self._add(data)

    def _add(self, data):
        base = self.next_base
        self.allocs.append((base, len(data), data))
        self.next_base += (len(data) + 255) // 256 * 256 + 256  # leave a gap between allocations
        return base

    def _find(self, addr, size):
        for base, n, data in self.allocs:
            if base <= addr and addr + size <= base + n:
                return data, addr - base
        raise MemoryFault("out-of-bounds global access at 0x%x (%d bytes)" % (addr, size))

    def load(self, addr, kind):
        data, off = self._find(addr, 4)
        return struct.unpack_from("<f" if kind == "f32" else "<i", data, off)[0]

    def store(self, addr, kind, value):
        data, off = self._find(addr, 4)
        struct.pack_into("<f" if kind == "f32" else "<i", data, off, value)

    def read_f32(self, base):
        for b, n, data in self.allocs:
            if b == base:
                return list(struct.unpack("<%df" % (n // 4), data))
        raise KeyError(base)

    def read_s32(self, base):
        for b, n, data in self.allocs:
            if b == base:
                return list(struct.unpack("<%di" % (n // 4), data))
        raise KeyError(base)


class Kernel:
    def __init__(self, name, params, instrs, labels):
        self.name, self.params, self.instrs, self.labels = name, params, instrs, labels


_ENTRY = re.compile(r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)\s*\{(.*?)\n\}", re.S)


def parse_ptx(text):
    kernels = {}
    for m in _ENTRY.finditer(text):
        name, params_txt, body = m.group(1), m.group(2), m.group(3)
        params = re.findall(r"\.param\s+\.(\w+)\s+(\w+)", params_txt)
        instrs, labels = [], {}
        for raw in body.split("\n"):
            s = raw.strip()
            if not s or s.startswith(".reg") or s.startswith("//"):
                continue
            if s.endswith(":"):
                labels[s[:-1]] = len(instrs)
                continue
            if not s.endswith(";"):
                raise ValueError("cannot parse line: " + raw)
            s = s[:-1].strip()
            pred = None
            if s.startswith("@"):
                guard, s = s.split(None, 1)
                neg = guard.startswith("@!")
                pred = (guard[2:] if neg else guard[1:], neg)
            op, _, rest = s.partition(" ")
            operands = [x.strip() for x in rest.split(",")] if rest else []
            instrs.append((pred, op, operands))
        kernels[name] = Kernel(name, params, instrs, labels)
    return kernels


def _special(name, tid, bid, bdim, gdim):
    kind, axis = name[1:].split(".")
    a = "xyz".index(axis)
    return {"tid": tid, "ctaid": bid, "ntid": bdim, "nctaid": gdim}[kind][a]


def _imm(text):
    if text.startswith("0f") or text.startswith("0F"):
        return struct.unpack("<f", struct.pack("<I", int(text[2:], 16)))[0]
    return int(text)


def run_thread(k, params, mem, tid, bid, bdim, gdim, max_steps=2_000_000):
    regs = {}

    def val(t):
        if t in regs:
            return regs[t]
        if t.startswith("%") and "." in t:
            return _special(t, tid, bid, bdim, gdim)
        if t.startswith("%"):
            raise RuntimeError("read of uninitialised register " + t)
        return _imm(t)

    pc, steps = 0, 0
    n = len(k.instrs)
    while pc < n:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("thread exceeded step limit (infinite loop?)")
        pred, op, o = k.instrs[pc]
        pc += 1
        if pred is not None:
            run_it = (not regs[pred[0]]) if pred[1] else bool(regs[pred[0]])
            if not run_it:
                continue
        parts = op.split(".")
        base = parts[0]
        ty = parts[-1]

        if base == "ret":
            return
        if base == "bra":
            pc = k.labels[o[0]]
        elif op.startswith("ld.param"):
            idx = [pn for _, pn in k.params].index(o[1][1:-1])
            regs[o[0]] = params[idx]
        elif op.startswith("cvta"):
            regs[o[0]] = val(o[1])
        elif base == "mov":
            regs[o[0]] = val(o[1])
        elif op.startswith("ld.global"):
            regs[o[0]] = mem.load(val(o[1][1:-1]), ty)
        elif op.startswith("st.global"):
            mem.store(val(o[0][1:-1]), ty, val(o[1]))
        elif op == "mul.wide.s32":
            regs[o[0]] = val(o[1]) * val(o[2])
        elif op == "add.s64":
            regs[o[0]] = val(o[1]) + val(o[2])
        elif ty == "s32" and base in ("add", "sub", "mul", "div", "rem", "neg") and not op.startswith("setp"):
            a = val(o[1])
            if base == "neg":
                r = -a
            else:
                b = val(o[2])
                if base == "add": r = a + b
                elif base == "sub": r = a - b
                elif base == "mul": r = a * b
                elif base == "div":
                    q = abs(a) // abs(b)
                    r = q if (a >= 0) == (b >= 0) else -q
                else:  # rem: sign follows dividend
                    r = abs(a) % abs(b)
                    r = r if a >= 0 else -r
            regs[o[0]] = wrap32(r)
        elif ty == "f32" and base in ("add", "sub", "mul", "div", "neg"):
            a = val(o[1])
            if base == "neg":
                r = -a
            else:
                b = val(o[2])
                if base == "add": r = a + b
                elif base == "sub": r = a - b
                elif base == "mul": r = a * b
                else:
                    if b == 0:
                        r = math.nan if (a == 0 or math.isnan(a)) else math.copysign(math.inf, a) * math.copysign(1, b)
                    else:
                        r = a / b
            regs[o[0]] = f32(r)
        elif base == "setp":
            cmp_ = parts[1]
            a, b = val(o[1]), val(o[2])
            regs[o[0]] = {
                "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b,
                "eq": a == b, "ne": a != b, "neu": a != b,
            }[cmp_]
        elif op == "not.pred":
            regs[o[0]] = not regs[o[1]]
        elif op == "and.pred":
            regs[o[0]] = bool(regs[o[1]] and regs[o[2]])
        elif op == "or.pred":
            regs[o[0]] = bool(regs[o[1]] or regs[o[2]])
        elif op == "cvt.rn.f32.s32":
            regs[o[0]] = f32(float(val(o[1])))
        elif op == "cvt.rzi.s32.f32":
            x = val(o[1])
            if math.isnan(x):
                regs[o[0]] = 0
            else:
                regs[o[0]] = max(-(1 << 31), min((1 << 31) - 1, int(x))) if math.isfinite(x) else (
                    (1 << 31) - 1 if x > 0 else -(1 << 31))
        else:
            raise NotImplementedError("simulator does not implement: " + op)


def launch(kernel, args, mem, grid, block):
    """args: list of ints (pointers/ints) or floats, in kernel parameter order."""
    gx, gy, gz = (list(grid) + [1, 1])[:3]
    bx, by, bz = (list(block) + [1, 1])[:3]
    params = []
    for (ty, _), a in zip(kernel.params, args):
        params.append(f32(a) if ty == "f32" else int(a))
    for bzi in range(gz):
        for byi in range(gy):
            for bxi in range(gx):
                for tzi in range(bz):
                    for tyi in range(by):
                        for txi in range(bx):
                            run_thread(kernel, params, mem, (txi, tyi, tzi), (bxi, byi, bzi), (bx, by, bz), (gx, gy, gz))
