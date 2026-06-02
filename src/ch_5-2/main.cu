#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Common/DS_timer.h"

#define NUM_DATA 1024 * 1024 * 10

__global__ void addVector(int* _dResult, const int* _dSrc1, const int* _dSrc2, int _size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < _size) {
    _dResult[idx] = _dSrc1[idx] + _dSrc2[idx];
  }
}

__global__ void add2DMatrix(int* _dResult, const int* _dSrc1, const int* _dSrc2) {
  uint32_t col = threadIdx.x;
  uint32_t row = threadIdx.y;
  uint32_t idx = row * blockDim.x + col;

  _dResult[idx] = _dSrc1[idx] + _dSrc2[idx];
}

int main(void) {
	// Set timer
	DS_timer timer(5);
	timer.setTimerName(0, (char*)"CUDA Total");
	timer.setTimerName(1, (char*)"Computation(Kernel)");
	timer.setTimerName(2, (char*)"Data Trans. : Host -> Device");
	timer.setTimerName(3, (char*)"Data Trans. : Device -> Host");
	timer.setTimerName(4, (char*)"VecAdd on Host");
	timer.initTimers();

  size_t memSize = sizeof(int) * NUM_DATA;
  int* src1 = new int[NUM_DATA];
  int* src2 = new int[NUM_DATA];
  int* result = new int[NUM_DATA];
  int* result_cpu = new int[NUM_DATA];

  for (int i = 0; i < NUM_DATA; i++) {
    src1[i] = rand() % 10;
    src2[i] = rand() % 10;
  }

  int* dSrc1;
  int* dSrc2;
  int* dResult;

  // Vector sum on CPU
  timer.onTimer(4);
  for (int i = 0; i < NUM_DATA; i++) {
    result_cpu[i] = src1[i] + src2[i];
  }
  timer.offTimer(4);

  cudaMalloc(&dSrc1, memSize);
  cudaMalloc(&dSrc2, memSize);
  cudaMalloc(&dResult, memSize);

  cudaMemset(dSrc1, 0, memSize);
  cudaMemset(dSrc2, 0, memSize);
  cudaMemset(dResult, 0, memSize);

  // CUDA Total timer start
  timer.onTimer(0);

  // Data copy from host to device
  timer.onTimer(2);
  cudaMemcpy(dSrc1, src1, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(dSrc2, src2, memSize, cudaMemcpyHostToDevice);
  timer.offTimer(2);

  // Kernel computation
  timer.onTimer(1);
  dim3 dimGrid(static_cast<uint32_t>(ceil(static_cast<float>(NUM_DATA) / 256)), 1, 1);
  dim3 dimBlock(256, 1, 1);
  addVector<<<dimGrid, dimBlock>>>(dResult, dSrc1, dSrc2, NUM_DATA);
  cudaDeviceSynchronize();  // Wait for the kernel to finish
  timer.offTimer(1);

  // Copy result back to host
  timer.onTimer(3);
  cudaMemcpy(result, dResult, memSize, cudaMemcpyDeviceToHost);
  timer.offTimer(3);

  // CUDA Total timer end
  timer.offTimer(0);

  cudaFree(dSrc1);
  cudaFree(dSrc2);
  cudaFree(dResult);

  timer.printTimer();

  // Check results
  bool compare = true;
  for (int i = 0; i < NUM_DATA; i++) {
    if (result_cpu[i] != result[i]) {
      printf("Mismatch at index %d: CPU result = %d, GPU result = %d\n", i, result_cpu[i], result[i]);
      compare = false;
    }
  }

  if (compare) {
    printf("CPU and GPU results match!\n");
  }

  delete[] src1;
  delete[] src2;
  delete[] result;
  delete[] result_cpu;

  return 0;
}

