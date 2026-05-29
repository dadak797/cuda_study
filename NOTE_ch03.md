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