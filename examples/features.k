// Language feature tour: early return, unary minus, ||, !, int wrap-around, int div/rem, casts.
kernel features(int* out, float* fout, int n) {
    int i = threadIdx.x;
    if (i >= n) {
        return;
    }
    int v = -i;
    if (i % 3 == 0 || (i > 5 && i < 8)) {
        v = v + 100;
    }
    if (!(i < 2)) {
        v = v * 2;
    }
    int w = 2147483647 + i;      // wraps around for i >= 1, like C on a GPU
    out[i] = v;
    out[i + n] = w / 7 - w % 7;
    float f = (float)i / 2;
    fout[i] = -f * 3.0;
    out[i + 2 * n] = (int)(f * 3.0) - 5;
}

// Two kernels in one file are fine.
kernel fill(int* out, int value, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = value;
    }
}
