// Short-circuit &&: a[i] is only loaded when i < n, so threads past the end never read out of bounds.
// (out must be zero-initialised by the caller.)
kernel count_positive(float* a, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] > 0.0) {
        out[i] = 1;
    }
}

// Int math: Collatz step count with a while loop (capped at 1000 steps).
kernel collatz(int* start, int* steps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int x = start[i];
        int count = 0;
        while (x != 1 && count < 1000) {
            if (x % 2 == 0) {
                x = x / 2;
            } else {
                x = 3 * x + 1;
            }
            count++;
        }
        steps[i] = count;
    }
}
