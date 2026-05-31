#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 1024

__global__ void addVector(int* _dResult, const int* _dSrc1, const int* _dSrc2) {
  int idx = threadIdx.x;
  _dResult[idx] = _dSrc1[idx] + _dSrc2[idx];
}

int main(void) {
  size_t memSize = sizeof(int) * NUM_DATA;
  int* src1 = new int[NUM_DATA];
  int* src2 = new int[NUM_DATA];
  int* result = new int[NUM_DATA];
  int* result_cpu = new int[NUM_DATA];

  int* dSrc1;
  int* dSrc2;
  int* dResult;

  cudaMalloc(&dSrc1, memSize);
  cudaMalloc(&dSrc2, memSize);
  cudaMalloc(&dResult, memSize);

  cudaMemset(dSrc1, 0, memSize);
  cudaMemset(dSrc2, 0, memSize);
  cudaMemset(dResult, 0, memSize);

  for (int i = 0; i < NUM_DATA; i++) {
    src1[i] = rand() % 10;
    src2[i] = rand() % 10;
  }

  cudaMemcpy(dSrc1, src1, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(dSrc2, src2, memSize, cudaMemcpyHostToDevice);

  addVector<<<1, NUM_DATA>>>(dResult, dSrc1, dSrc2);

  cudaMemcpy(result, dResult, memSize, cudaMemcpyDeviceToHost);

  bool compare = true;
  for (int i = 0; i < NUM_DATA; i++) {
    result_cpu[i] = src1[i] + src2[i];
    if (result_cpu[i] != result[i]) {
      compare = false;
      break;
    }
  }

  if (compare) {
    printf("CPU and GPU results match!\n");
  } else {
    printf("CPU and GPU results do not match!\n");
  }

  delete[] src1;
  delete[] src2;
  delete[] result;
  delete[] result_cpu;

  cudaFree(dSrc1);
  cudaFree(dSrc2);
  cudaFree(dResult);

  return 0;
}

