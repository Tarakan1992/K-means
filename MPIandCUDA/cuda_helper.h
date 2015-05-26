#ifndef __CUDA_HELPER_H
#define __CUDA_HELPER_H

#include <cuda.h>
#include <stdio.h>

cudaDeviceProp getDeviceProperties(int device_id)
{
	cudaDeviceProp dp;
	
	cudaGetDeviceProperties(&dp, device_id);
	
	return dp;
}

void printDeviceProperties(cudaDeviceProp dp, int device_id)
{
	printf("Device %d\n", device_id);
	printf("Compute capability     : %d.%d\n", dp.major, dp.minor);
	printf("Name                   : %s\n", dp.name);
	printf("Total Global Memory    : %lu\n", dp.totalGlobalMem);
	printf("Shared memory per block: %lu\n", dp.sharedMemPerBlock);
	printf("Registers per block    : %d\n", dp.regsPerBlock);
	printf("Warp size              : %d\n", dp.warpSize);
	printf("Max threads per block  : %d\n", dp.maxThreadsPerBlock);
	printf("Multiprocessor Count   : %d\n", dp.multiProcessorCount);
	printf("Max Threads Dim        : %d %d %d\n",
								dp.maxThreadsDim[0],
								dp.maxThreadsDim[1],
								dp.maxThreadsDim[2]);
	printf("Max Grid Size          : %d %d %d\n",
								dp.maxGridSize[0],
								dp.maxGridSize[1],
								dp.maxGridSize[2]);
}

void printAllDevicesProperties()
{
	int device_count;

	cudaGetDeviceCount(&device_count);

	for(int i = 0; i < device_count; i++)
	{
		printDeviceProperties(getDeviceProperties(i), i);
	}
}

#endif // __CUDA_HELPER_H