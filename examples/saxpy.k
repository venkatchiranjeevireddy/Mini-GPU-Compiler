// y[i] = alpha * x[i] + y[i]
kernel saxpy(int n, float alpha, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}
