# CUDA keywords
- __host__ : called from CPU, executed on CPU
- __device__ : called from GPU, executed on GPU
- __global__ : called from CPU, executed on GPU

# Launch the kernel
```
helloCUDA<<<1, 10>>>();
```
- Launch the kernel with 1 block and 10 threads