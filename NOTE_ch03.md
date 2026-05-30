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

