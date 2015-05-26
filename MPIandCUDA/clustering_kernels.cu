#include <cuda.h>
#include <float.h>

__global__ void clusteringKernel(
	float* items, 
	float* centers, 
	int* clustersIds, 
	int itemsCount, 
	int centersCount,
	int paramsCount,
	float* newCenters, 
	int* itemsPerCluster)
{
	int itemId = blockDim.x * blockIdx.x + threadIdx.x;

	if(itemId < itemsCount)
	{
		float minDistance = FLT_MAX;
		float sum;

		for(int i = 0; i < centersCount; i++)
		{
			sum = 0.0;

			for(int j = 0; j < paramsCount; j++)
			{
				sum = sum + (items[itemId * paramsCount + j] - centers[i * paramsCount + j]) 
				* (items[itemId * paramsCount + j] - centers[i * paramsCount + j]);
			}

			if(minDistance >= sum)
			{
				minDistance = sum;
				clustersIds[itemId] = i;
			}
		}


		for(int i = 0; i < paramsCount; i++)
		{
			atomicAdd(newCenters + clustersIds[itemId] * paramsCount + i,
				items[itemId * paramsCount + i]);
		}

		atomicAdd(itemsPerCluster + clustersIds[itemId], 1);
	}
}