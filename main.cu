#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void printData(int* _dDataPtr) {
  printf("%d", _dDataPtr[threadIdx.x]);
}

__global__ void setData(int* _dDataPtr) {
  // threadIdx.x is internal variable that holds the thread index within the block.
  _dDataPtr[threadIdx.x] = 2;
}

int main(void) {
  // Initialize data on host
  int data[10] = { 0 };
  for (int i = 0; i < 10; i++) {
    data[i] = 1;
  }

  // Allocate memory on device and initialize it to zero
  int* dDataPtr;
  cudaMalloc(&dDataPtr, sizeof(int) * 10);
  cudaMemset(dDataPtr, 0, sizeof(int) * 10);

  printf("Data in device: ");
  printData<<<1, 10>>>(dDataPtr);

  // Copy data from host to device and print it
  cudaMemcpy(dDataPtr, data, sizeof(int) * 10, cudaMemcpyHostToDevice);
  printf("\nHost -> Device: ");
  printData<<<1, 10>>>(dDataPtr);

  // Set data in device
  setData<<<1, 10>>>(dDataPtr);

  // Copy data from device to host and print it
  cudaMemcpy(data, dDataPtr, sizeof(int) * 10, cudaMemcpyDeviceToHost);
  printf("\nDevice -> Host: ");
  for (int i = 0; i < 10; i++) {
    printf("%d", data[i]);
  }

  // Free device memory
  cudaFree(dDataPtr);

  return 0;
}
