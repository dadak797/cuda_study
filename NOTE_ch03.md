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

# CUDA 알고리즘의 성능 측정
- 커널 호출 시 디바이스에게 명령을 전달한 후 프로그램 흐름의 제어권을 바로 호스트에게 반환함. 즉 비동기적으로 커널을 호출함
- 성능을 측정하기 위해서는 동기화 함수(cudaDeviceSynchronize())를 이용하여 디바이스가 수행 중인 작업이 끝날 때까지 대기해야 함

```
// 시간 측정 시작
vecAdd<<<1, NUM_DATA>>>(da, db, dc);
cudaDeviceSynchronize();
// 시간 측정 종료
```
- CUDA 이벤트를 이용하면 더 정확하게 측정할 수 있음
- CUDA API의 호출은 순차적으로 실행되기 때문에, 호스트 코드를 디바이스 코드의 제어에만 사용하는 경우에는 동기화 함수를 호출하지 않아도 됨. 하지만 호스트와 디바이스 모두를 이용해 연산을 하는 경우에는 서로 순서를 맞추거나 정보를 주고 받을 때는 동기화가 필요함

### 데이터 전송 시간
- CPU에서 GPU 메모리 혹은 GPU에서 CPU 메모리로 데이터를 전송하는 것도 추가적으로 발생하는 작업이기 때문에 전송 시간도 성능 측정에 포함되어야 함

```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Common/DS_timer.h"

#define NUM_DATA 1024*1024

__global__ void addVector(int* _dResult, const int* _dSrc1, const int* _dSrc2) {
  int idx = threadIdx.x;
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
  addVector<<<1, NUM_DATA>>>(dResult, dSrc1, dSrc2);
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
      break;
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
```
- Timer를 이용하여 계산 시간을 측정
- 데이터의 크기가 작은 경우에는 CPU에서 작업을 하는 것이 더 빠름
- 데이터 복사 시간도 상당히 걸림
- 데이터의 크기가 커지면 두 연산 결과에 차이가 발생함 (threadIdx의 문제)
