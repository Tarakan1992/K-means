#include <float.h>

__device__ float euclid2(
	float* items, 
	float* centers, 
	int itemId, 
	int centerId, 
	int paramsCount)
{
	float sum = 0.0;

	for(int j = 0; j < paramsCount; j++)
	{
		sum = sum + (items[itemId * paramsCount + j] - centers[centerId * paramsCount + j]) 
		* (items[itemId * paramsCount + j] - centers[centerId * paramsCount + j]);
	}

	return sum;
}


__global__ void clusreringKernel(
	float* items, 
	float* deviceCenters,
	int* clustersIds,
	int itemsCount,
	int clustersCount, 
	int paramsCount)
{

	extern __shared__ float centers[];


	for (int i = threadIdx.x; i < clustersCount; i += blockDim.x) 
	{
        for (int j = 0; j < paramsCount; j++) 
        {
            centers[clustersCount * j + i] = deviceCenters[clustersCount * j + i];
        }
    }

    __syncthreads();

	int itemId = blockDim.x * blockIdx.x + threadIdx.x;


	if(itemId < itemsCount)
	{
		float minDistance = FLT_MAX;
		float distance;
		int index;

		for(int i = 0; i < clustersCount; i++)
		{
			distance = euclid2(items, centers, itemId, i, paramsCount);

			if(minDistance >= distance)
			{
				minDistance = distance;
				index = i;
			}
		}

		clustersIds[itemId] = index;
	}
}


__global__ void newCentersKernel(
	float* items,
	int* clustersIds,
	float* newCenters,
	int* itemsPerClusters,
	int itemsCount,
	int clustersCount,
	int paramsCount
	)
{
	int clusterId = blockIdx.x;

	for(int i = 0; i < itemsCount; i++)
	{
		if(clusterId == clustersIds[i])
		{
			for(int j = 0; j < paramsCount; j++)
			{
				newCenters[blockIdx.x * paramsCount + j] += items[i * paramsCount + j];
			}

			itemsPerClusters[clusterId] += 1;
		}
	}
}