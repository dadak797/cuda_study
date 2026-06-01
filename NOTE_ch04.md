# CUDA 스레드 계층

### 스레드 (Thread)
- 가장 낮은 계층
- 작성한 커널 코드는 모든 스레드에 공유됨
- 각 스레드가 독립적으로 커널 코드를 수행

### 워프 (Warp)
- 32개의 스레드를 하나로 묶은 것
- CUDA의 기본 수행 단위
- GPU의 SIMT 구조에서 멀티 스레드의 단위가 되는 것이 바로 워프
- 하나의 명령에 따라 32개의 스레드가 동시에 움직임

### 블록 (Block)
- 워프들의 집합
- 하나의 블록에 포함된 각 스레드는 자신만의 고유한 스레드 번호를 가짐 (동일한 블록 안에는 동일한 번호를 갖는 스레드가 없다는 의미)
- 서로 다른 블록에 포함된 스레드들은 같은 스레드 번호를 가질 수 있음
- 각 블록은 자신만의 고유한 블록 ID를 가지고 있으며, 원하는 스레드를 정확히 지칭하기 위해서는 블록 번호와 스레드 번호를 모두 사용해야 함
- 블록 내 스레드는 1차원, 2차원, 3차원 형태로 배치될 수 있음

### 그리드
- 가장 상위 단계
- 여러 개의 블록을 포함하는 블록들의 그룹
- 하나의 그리드에 포함된 블록들은 서로 다른 자신만의 고유한 블록 번호를 가짐
- 하나의 그리드는 하나의 커널 호출과 1:1 대응됨

# CUDA 스레드 계층을 위한 내장 변수들
- 그리드 및 블록의 형태, 각 스레드가 자신이 속한 블록 번호를 제공
- 내장 변수의 값은 커널이 실행될 때 결정됨

### gridDim
- 그리드의 형태 정보를 담고 있는 구조체
- x, y, z 멤버가 각 차원의 크기를 담고 있음
- 그리드의 x-차원 최대 길이는 2^32-1이고 y-와 z-차원은 65535임

### blockIdx
- 현재 스레드가 속한 블록의 번호를 담고 있는 구조체형 내장 변수

### blockDim
- 블록의 형태 정보를 담고 있는 구조체형 내장 변수
- x-, y-차원의 최대 크기는 1024이고, z-차원의 최대 크기는 64임

### threadIdx
- 블록 내에서 현재 스레드가 부여받은 스레드 번호를 담고 있는 구조체형 내장 변수
- 워프는 연속된 32개의 스레드로 구성됨

# CUDA 스레드 구조와 커널 호출

### 스레드 레이아웃 설정 및 커널 호출
```
Kernel<<<Grid Layout, Block Layout>>>
```
- <<<1, n>>>으로 설정하면 Grid의 크기 (1, 1, 1), Block의 크기 (n, 1, 1)을 사용하라는 뜻
- <<<(3, 2, 1), (6, 4, 2)>>> 같은 형태로 Grid와 Block의 크기를 전달할 수 있음
- CUDA가 지원하는 구조체 dim3을 이용하여 크기를 전달하는 것이 일반적임

```
dim3 dimGrid(4, 1, 1);
dim3 dimBlock(8, 1, 1);
kernel<<<dimGrid, dimBlock>>>();
```
- blockIdx.x가 [0, 3], threadIdx.x가 [0, 7]인 스레드 레이아웃을 생성함

### 스레드 레이아웃과 index 확인하기
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void checkIndex() {
  printf("threadIdx: (%d, %d, %d), blockIdx: (%d, %d, %d), blockDim: (%d, %d, %d), gridDim: (%d, %d, %d)\n",
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z);
}

int main() {
  dim3 dimBlock(3, 1, 1);
  dim3 dimGrid(2, 1, 1);

  printf("dimGrid.x=%d, dimGrid.y=%d, dimGrid.z=%d\n", dimGrid.x, dimGrid.y, dimGrid.z);
  printf("dimBlock.x=%d, dimBlock.y=%d, dimBlock.z=%d\n", dimBlock.x, dimBlock.y, dimBlock.z);

  checkIndex<<<dimGrid, dimBlock>>>();

  return 0;
}
```

출력 결과
```
dimGrid.x=2, dimGrid.y=1, dimGrid.z=1
dimBlock.x=3, dimBlock.y=1, dimBlock.z=1
threadIdx: (0, 0, 0), blockIdx: (0, 0, 0), blockDim: (3, 1, 1), gridDim: (2, 1, 1)
threadIdx: (1, 0, 0), blockIdx: (0, 0, 0), blockDim: (3, 1, 1), gridDim: (2, 1, 1)
threadIdx: (2, 0, 0), blockIdx: (0, 0, 0), blockDim: (3, 1, 1), gridDim: (2, 1, 1)
threadIdx: (0, 0, 0), blockIdx: (1, 0, 0), blockDim: (3, 1, 1), gridDim: (2, 1, 1)
threadIdx: (1, 0, 0), blockIdx: (1, 0, 0), blockDim: (3, 1, 1), gridDim: (2, 1, 1)
threadIdx: (2, 0, 0), blockIdx: (1, 0, 0), blockDim: (3, 1, 1), gridDim: (2, 1, 1)
```
- blockDim과 gridDim은 모든 스레드에서 공유함
- 한 block 안에서는 스레드 번호가 서로 다르지만, 다른 block에 있는 스레드 중에는 동일한 번호를 가진 스레드가 존재함
- 같은 스레드 번호를 가진 두 개의 스레드는 서로 다른 block을 가짐

# 큰 벡터에 대한 벡터 합 - 스레드 레이아웃
```
vecAdd<<<1, NUM_DATA>>>(da, db, dc);
```
- 블록의 x-차원의 최대 길이는 1024이며, 한 블록이 가질 수 있는 최대 스레드 수는 1024임
- 따라서, 데이터가 1024가 넘으면 블록의 개수를 늘려야 함

```
vecAdd<<<ceil(NUM_DATA/1024), NUM_DATA>>>(da, db, dc);
```
- 데이터가 1024가 넘어가면 올림을 통해, 블록의 개수를 추가해야 함
- 블록 내의 최대 스레드 수인 1024보다 큰 값을 이용하여 설정하면 계산이 아예 수행되지 않음
- 블록이 추가되면 블록의 인덱스를 고려하여 스레드 인덱싱을 해야 독립적인 id로 연산을 수행할 수 있음
