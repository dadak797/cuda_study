# CUDA 기초 메모리 API
### 디바이스(GPU) 메모리 할당
```
cudaError_t cudaMalloc(void** ptr, size_t size)
```
- ptr: 할당된 디바이스 메모리 시작 주소. 호스트(CPU)에서 디바이스 메모리에 직접 접근하여 사용할 수는 없음

### 디바이스 메모리 해제
```
cudaError_t cudaFree(void* ptr)
```

### 디바이스 메모리 초기화
```
cudaError_t cudaMemset(void* ptr, int value, size_t size)
```

### 에러 코드 확인
```
__host__ __device__ const char* cudaGetErrorName(cudaError_t error)
```
- enum 타입인 cudaError_t를 이용하여 에러 메시지를 확인할 수 있음
- 호스트와 디바이스 모두에서 사용할 수 있음

### 디바이스 메모리 확인
```
__host__ cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
```
- 사용 가능한 메모리 크기와 전체 메모리 크기를 알려줌

### 호스트-디바이스 메모리 복사
```
__host__ cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
```
cudaMemcpyKind에는 5가지 옵션이 있음
- cudaMemcpyHostToHost
- cudaMemcpyHostToDevice
- cudaMemcpyDeviceToHost
- cudaMemcpyDeviceToDevice
- cudaMemcpyDefault: 포인터를 보고 추론 (unified virtual addressing을 지원하는 시스템에서 사용 가능함. 추천하지 않음)

Unified memory: CPU 메모리와 GPU 메모리를 하나의 논리적 주소 공간으로 사용할 수 있는 기술. 임베디드 GPU를 사용하는 경우에 필요함. 자주 사용하면 성능이 떨어지니 명시적인 옵션을 사용하는 것이 좋음

cudaMemcpy2D(): 2차원 데이터의 복사
cudaMemcpy2D(): 3차원 데이터의 복사
cudaMemcpyAsync(): 비동기 데이터 복사

# 벡터 더하기 프로그램
```
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
```
- 루프를 돌지 않고, 각 thread의 index(threadIdx.x)를 이용하여 벡터의 각 요소별 연산을 수행함