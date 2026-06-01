#pragma once

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
