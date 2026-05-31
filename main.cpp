#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 1024

int main(void) {
  size_t memSize = sizeof(int) * NUM_DATA;
  int* src1 = new int[NUM_DATA];
  int* src2 = new int[NUM_DATA];
  int* result = new int[NUM_DATA];

  memset(src1, 0, memSize);
  memset(src2, 0, memSize);
  memset(result, 0, memSize);

  for (int i = 0; i < NUM_DATA; i++) {
    src1[i] = rand() % 10;
    src2[i] = rand() % 10;
  }

  for (int i = 0; i < NUM_DATA; i++) {
    result[i] = src1[i] + src2[i];
  }

  delete[] src1;
  delete[] src2;
  delete[] result;

  return 0;
}
