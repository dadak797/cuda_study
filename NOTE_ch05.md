# 스레드 레이아웃과 인덱싱
- 내장 인덱스를 이용하여 다른 블록에서 중복되지 않는 전역 인덱스를 만들기

### 1024보다 큰 벡터의 합 구하기
- 다른 블록에서 중복되지 않는 인덱스를 가지려면 블록의 크기와 블록의 인덱스를 이용하면 됨
```
int tID = blockIdx.x * blockDim.x + threadIdx.x;
```

### 블록의 크기를 벗어나는 경우의 예외 처리 하기
```
__global__ void addVector(int* _dResult, const int* _dSrc1, const int* _dSrc2, int _size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < _size) {
    _dResult[idx] = _dSrc1[idx] + _dSrc2[idx];
  }
}
```

```
dim3 dimGrid(static_cast<uint32_t>(ceil(static_cast<float>(NUM_DATA) / 256)), 1, 1);
dim3 dimBlock(256, 1, 1);
addVector<<<dimGrid, dimBlock>>>(dResult, dSrc1, dSrc2, NUM_DATA);
```

## 블록 내 스레드의 전역 번호

### 1차원 블록
- threadIdx.x가 스레드의 전역 번호와 동일함

### 2차원 블록
```
2D_BLOCK_TID = (blockDim.x * threadIdx.y + threadIdx.x)
```

### 3차원 블록
```
TID_IN_BLOCK = (blockDim.x * blockDim.y * threadIdx.z + 2D_BLOCK_TID)
```

## 그리드 내 스레드의 전역 번호
- 앞에서 계산한 블록 내 스레드의 전역 번호(TID_IN_BLOCK)를 이용하여 구현함

### 1차원 그리드
- 블록 하나에 속한 스레드의 개수
```
NUM_THREAD_IN_BLOCK = (blockDim.z * blockDim.y * blockDim.x)
1D_GRID_TID = (blockIdx.x * NUM_THREAD_IN_BLOCK) + TID_IN_BLOCK
```

### 2차원 그리드
```
2D_GRID_TID = (blockIdx.y * (gridDim.x * NUM_THREAD_IN_BLOCK)) + 1D_GRID_TID
```

### 3차원 그리드
```
GLOBAL_TID = (blockIdx.z * (gridDim.y * gridDim.x * NUM_THREAD_IN_BLOCK)) + 2D_GRID_TID
```

### 스레드 인덱싱 definitions
```
// Block ID
#define BID_X blockIdx.x
#define BID_Y blockIdx.y
#define BID_Z blockIdx.z

// Thread ID
#define TID_X threadIdx.x
#define TID_Y threadIdx.y
#define TID_Z threadIdx.z

// Dimension of a grid
#define GDIM_X gridDim.x
#define GDIM_Y gridDim.y
#define GDIM_Z gridDim.z

// Dimension of a block
#define BDIM_X blockDim.x
#define BDIM_Y blockDim.y
#define BDIM_Z blockDim.z

#define TID_IN_BLOCK (TID_Z * (BDIM_Y * BDIM_X) + TID_Y * BDIM_X + TID_X)
#define NUM_THREADS_IN_BLOCK (BDIM_Z * BDIM_Y * BDIM_X)

#define GRID_1D_TID (BID_X * NUM_THREADS_IN_BLOCK + TID_IN_BLOCK)
#define GRID_2D_TID (BID_Y * (GDIM_X * NUM_THREADS_IN_BLOCK) + GRID_1D_TID)
#define GLOBAL_TID (BID_Z * (GDIM_Y * GDIM_X * NUM_THREADS_IN_BLOCK) + GRID_2D_TID)
```

## 행렬 데이터 인덱싱
- 2차원 스레드 번호를 사용하여 각 스레드가 행렬의 담당 원소를 가리키게 함
```
col = threadIdx.x
row = threadIdx.y
```
```
index (row, col) = row * blockDim.x + col
                 = threadIdx.y * blockDim.x + threadIdx.x
```

- 행렬 더하기 함수
```
__global__ void add2DMatrix(int* _dResult, const int* _dSrc1, const int* _dSrc2) {
  uint32_t col = threadIdx.x;
  uint32_t row = threadIdx.y;
  uint32_t idx = row * blockDim.x + col;

  _dResult[idx] = _dSrc1[idx] + _dSrc2[idx];
}
```
- 커널 호출
```
dim3 blockDim(COL_SIZE, ROW_SIZE);
add2DMatrix<<<1, blockDim>>>(dResult, dSrc1, dSrc2);
```
