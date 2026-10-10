"""
fake_cuda.py - a stand-in for libcuda that executes kernels with the PTX simulator.

It lets the test-suite exercise host/run_gpu.py's argument marshalling (ctypes pointers, parameter sizes and
ordering, device-pointer bookkeeping) on a machine without a GPU.  It is NOT a substitute for running
host/run_gpu.py on real hardware.
"""
import ctypes

import ptx_sim as sim


class _Fn:
    def __init__(self, impl):
        self.impl = impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *a):
        return self.impl(*a)


def _set(byref_obj, value):
    byref_obj._obj.value = value


class FakeCuda:
    def __init__(self):
        self.mem = sim.Memory()
        self.modules = {}
        self.functions = {}
        for name in dir(self):
            if name.startswith("cu") and callable(getattr(self, name)):
                setattr(self, name, _Fn(getattr(self, name)))

    # --- driver API (names match libcuda) --------------------------------------------------
    def cuInit(self, flags):
        return 0

    def cuDeviceGet(self, pdev, ordinal):
        _set(pdev, 0)
        return 0

    def cuDeviceGetName(self, buf, n, dev):
        buf.value = b"Fake GPU (PTX simulator)"
        return 0

    def cuDevicePrimaryCtxRetain(self, pctx, dev):
        _set(pctx, 1)
        return 0

    def cuCtxSetCurrent(self, ctx):
        return 0

    def cuModuleLoadDataEx(self, pmod, image, nopts, opts, vals):
        assert isinstance(image, (bytes, bytearray)) and image.endswith(b"\0"), "PTX image must be NUL-terminated"
        kernels = sim.parse_ptx(image[:-1].decode())
        mid = len(self.modules) + 1
        self.modules[mid] = kernels
        _set(pmod, mid)
        return 0

    def cuModuleGetFunction(self, pfn, mod, name):
        mid = mod.value if hasattr(mod, "value") else mod
        fid = len(self.functions) + 1
        self.functions[fid] = self.modules[mid][name.decode()]
        _set(pfn, fid)
        return 0

    def cuMemAlloc_v2(self, pdptr, nbytes):
        _set(pdptr, self.mem._add(bytearray(nbytes)))
        return 0

    def cuMemFree_v2(self, dptr):
        return 0

    def cuMemcpyHtoD_v2(self, dst, src, nbytes):
        src_addr = src.value if hasattr(src, "value") else src
        data, off = self.mem._find(dst, nbytes)
        data[off:off + nbytes] = ctypes.string_at(src_addr, nbytes)
        return 0

    def cuMemcpyDtoH_v2(self, dst, src, nbytes):
        dst_addr = dst.value if hasattr(dst, "value") else dst
        data, off = self.mem._find(src, nbytes)
        ctypes.memmove(dst_addr, bytes(data[off:off + nbytes]), nbytes)
        return 0

    def cuLaunchKernel(self, fn, gx, gy, gz, bx, by, bz, shared, stream, params, extra):
        fid = fn.value if hasattr(fn, "value") else fn
        kernel = self.functions[fid]
        args = []
        for i, (ty, _) in enumerate(kernel.params):
            addr = params[i]  # address of the i-th argument value
            ctype = {"u64": ctypes.c_uint64, "u32": ctypes.c_int, "f32": ctypes.c_float}[ty]
            args.append(ctype.from_address(addr).value)
        sim.launch(kernel, args, self.mem, (gx, gy, gz), (bx, by, bz))
        return 0

    def cuCtxSynchronize(self):
        return 0
