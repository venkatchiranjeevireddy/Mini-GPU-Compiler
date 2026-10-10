// Grid-stride loop: any grid size can process any n.
// Also uses compound assignment, a cast, and a constant expression the optimizer folds.
kernel scale_stride(float* data, int n, float factor) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        data[i] = data[i] * factor + (float)(2 * 3 - 6);
    }
}
