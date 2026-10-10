// out[i] = max(in[i], 0) written with if/else, plus a leaky variant
kernel relu(float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (in[i] > 0.0) {
            out[i] = in[i];
        } else {
            out[i] = 0.0;
        }
    }
}

kernel leaky_relu(float* in, float* out, int n, float slope) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        if (v < 0.0) {
            v = v * slope;
        }
        out[i] = v;
    }
}
