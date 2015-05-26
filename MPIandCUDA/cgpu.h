#ifndef __CGPU_H
#define __CGPU_H


#include "cuda_helper.h"
#include "clustering.h"
#include "clustering_kernels.h"
#include <float.h>
#include <mpi.h>


#define min(X, Y) (((X) < (Y)) ? (X) : (Y))

int getBlockSize(int itemsCount, cudaDeviceProp dp)
{
	return min(itemsCount, dp.maxThreadsPerBlock / 4);
}

int getGridSize(int itemsCount, int centersCount, int blockSize)
{    
    int gridX = itemsCount / blockSize;
    
    gridX += (itemsCount % blockSize) == 0 ? 0 : 1;   

    return gridX;
}

void runGPUClustering(cselection* selections, ccenters* c, ccenters* cs, int clustersCount, int paramsCount, int deviceCount)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);



	float* d_centers;
	cudaMalloc((void**)&d_centers, clustersCount * paramsCount * sizeof(float));
	cudaMemcpy(d_centers, c->centers, clustersCount * paramsCount * sizeof(float), cudaMemcpyHostToDevice);

  	memset(c->newCenters, 0.0, clustersCount * paramsCount * sizeof(float));
  	memset(c->itemsPerClusters, 0, clustersCount * sizeof(float));
	

	for (int i = 0; i < deviceCount; i++)
	{

		cudaSetDevice(i);

		cudaMemsetAsync(cs[i].d_newCenters, 0.0,
			clustersCount * paramsCount * sizeof(float), selections[i].stream);

		cudaMemsetAsync(cs[i].d_itemsPerClusters, 0,
			clustersCount * sizeof(int), selections[i].stream);

		cudaDeviceProp dp = getDeviceProperties(i);

		int blockSize = getBlockSize(selections[i].itemsCount, dp);
		int gridSize = getGridSize(selections[i].itemsCount, clustersCount, blockSize);

		//printf("BLOCK SIZE: %d\n", blockSize);
		//printf("GRID SIZE: %d\n", gridSize);

		clusteringKernel<<<gridSize, blockSize, dp.sharedMemPerBlock, selections[i].stream>>>(
			selections[i].d_items, 
			d_centers, 
			selections[i].d_clustersIds, 
			selections[i].itemsCount, 
			clustersCount, 
			paramsCount,
			cs[i].d_newCenters,
			cs[i].d_itemsPerClusters);

		cudaMemcpyAsync(cs[i].newCenters, cs[i].d_newCenters,
	    	clustersCount * paramsCount * sizeof(float), cudaMemcpyDeviceToHost, selections[i].stream);

		cudaMemcpyAsync(cs[i].itemsPerClusters, cs[i].d_itemsPerClusters,
	    	clustersCount * sizeof(int), cudaMemcpyDeviceToHost, selections[i].stream);

		cudaMemcpyAsync(selections[i].clustersIds, selections[i].d_clustersIds,
  			selections[i].itemsCount * sizeof(int), cudaMemcpyDeviceToHost, selections[i].stream);

		
	}

	printf("ALL OK!\n");

   	for (int i = 0; i < deviceCount; i++)
    {
       	cudaSetDevice(i);
        cudaStreamSynchronize(selections[i].stream);

        for(int j = 0; j < clustersCount; j++)
        {
        	c->itemsPerClusters[j] += cs[i].itemsPerClusters[j];

        	for(int k = 0; k < paramsCount; k++)
        	{
        		c->newCenters[j * paramsCount + k] += cs[i].newCenters[j * paramsCount + k];
        	}
        }
    }

	printf("ALL OK!\n");



	// if(rank == 1)
	// {
	// 	for(int i = 0; i < deviceCount; i++)
	// 	{
			// for(int j = 0; j < selections[i].itemsCount; j++)
			// {
			// 	printf("cluster id: %d\n", selections[i].clustersIds[j]);
			// 	for(int k = 0; k < paramsCount; k++)
			// 	{
			// 		printf("%f ", selections[i].items[j * paramsCount + k]);
			// 	}
			// 	printf("\n");
			// }

			// printf("\n");

	// 		for(int j = 0; j < clustersCount; j++)
	// 		{
	// 			printf("itemsCount: %d\n", cs[i].itemsPerClusters[j]);
	// 			for(int k = 0; k < paramsCount; k++)
	// 			{
	// 				printf("%f ", cs[i].newCenters[j * paramsCount + k]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// }

	cudaFree(d_centers);
}

void initGPUClustering(cselection* selections, ccenters* cs, int deviceCount)
{
	for(int i = 0; i < deviceCount; i++)
	{
		cudaSetDevice(i);
		cudaDeviceReset();
		cudaStreamCreate(&(selections[i].stream));
		copySelectionToDevice(&(selections[i]));
		copyCentersToDevice(&(cs[i]));
	}
}

void freeGPUMemory(cselection* selections, ccenters* cs, int deviceCount)
{
	for(int i = 0; i < deviceCount; i++)
	{
		cudaSetDevice(i);
		freeSelectionOnDevice(selections + i);
		freeCentersOnDevice(cs + i);
	}

	cudaDeviceReset();
}


#endif // __CGPU_H