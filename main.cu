#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void checkDeviceMemory() {
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  printf("Device memory (free/total) = %lld / %lld bytes\n", free, total);
}

int main(void) {
  int* dDataPtr;
  cudaError_t errorCode;

  checkDeviceMemory();

  // Allocate 4MB of device memory
  errorCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024);
  printf("cudaMalloc - %s\n", cudaGetErrorString(errorCode));
  checkDeviceMemory();

  errorCode = cudaMemset(dDataPtr, 0, sizeof(int) * 1024 * 1024);
  printf("cudaMemset - %s\n", cudaGetErrorString(errorCode));
  checkDeviceMemory();

  errorCode = cudaFree(dDataPtr);
  printf("cudaFree - %s\n", cudaGetErrorString(errorCode));
  checkDeviceMemory();

  return 0;
}