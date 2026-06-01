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

