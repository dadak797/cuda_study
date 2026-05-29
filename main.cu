#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// CUDA keywords
// __host__ : called from CPU, executed on CPU
// __device__ : called from GPU, executed on GPU
// __global__ : called from CPU, executed on GPU
__global__ void helloCUDA(void) {
  printf("Hello CUDA from GPU!\n");
}

int main(void) {
  printf("Hello GPU from CPU!\n");
  // Launch the kernel with 1 block and 10 threads
  helloCUDA<<<1, 10>>>();
  
  return 0;
}