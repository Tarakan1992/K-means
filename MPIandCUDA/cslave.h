#ifndef __CSLAVE_H
#define __CSLAVE_H

#include <cuda.h>

#include "csystem.h"
#include "cgpu.h"
#include "clustering.h"
#include <mpi.h>

void reciveItems(cselection* selecton)
{
	MPI_Status status;

	MPI_Recv((void*)(selecton->items), selecton->itemsCount * selecton->paramsCount, 
		MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
}

void reciveItemsCount(int* itemsCount)
{
	MPI_Status status;

	MPI_Recv((void*)itemsCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
}

void reciveCenters(float* centers, int clustersCount, int paramsCount)
{
	MPI_Status status;

	MPI_Recv((void*)centers, clustersCount * paramsCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
}

void reciveEOC(int* EOC)
{
	MPI_Status status;

	MPI_Recv((void*)EOC, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
}

void sendClustersIds(cselection selecton)
{
	MPI_Send((void*)selecton.clustersIds, selecton.itemsCount, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

void sendNewCenters(ccenters c, int clustersCount, int paramsCount)
{
	MPI_Send((void*)c.newCenters, clustersCount * paramsCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	MPI_Send((void*)c.itemsPerClusters, clustersCount, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

void runSlaveClusteringCycle(cselection* selections, int deviceCount, int clustersCount, int paramsCount)
{
	int EOC;
	//int rank;
	ccenters c = createCenters(clustersCount, paramsCount);
	ccenters* cs = distributeCenters(c, deviceCount);

	initGPUClustering(selections, cs, deviceCount);

	//MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	do
	{
		reciveCenters(c.centers, clustersCount, paramsCount);

		runGPUClustering(selections, &c, cs, clustersCount, paramsCount, deviceCount);
		
		// if(rank == 1)
		// {
		// 	printf("Slave centers send: \n");
		// 	for(int i = 0; i < clustersCount; i++)
		// 	{
				
		// 		printFloatMassive(c.centers + i * paramsCount, paramsCount);
		// 		printf("%d \n", c.itemsPerClusters);
		// 	}
		// 	printf("\n\n");
		// }

		sendNewCenters(c, clustersCount, paramsCount);

		reciveEOC(&EOC);
	}while(!EOC);

	freeGPUMemory(selections, cs, deviceCount);
}

void runSlaveProcess(csystem dsystem)
{

	int itemsCount;
	int deviceCount;
	int paramsCount = dsystem.paramsCount;

	reciveItemsCount(&itemsCount);

	cselection mainSelection = createSelection(itemsCount, paramsCount);

	reciveItems(&mainSelection);

	// if(dsystem.rank == 1)
	// {
	// 	for(int i = 0; i < itemsCount; i++)
	// 	{
	// 		printFloatMassive(mainSelection.items + i * paramsCount, paramsCount);
	// 	}
	// 	printf("\n");
	// }

	cudaGetDeviceCount(&deviceCount);

	cselection* selections = distributeSelection(mainSelection, deviceCount);

	runSlaveClusteringCycle(selections, deviceCount, dsystem.clustersCount, paramsCount);

	sendClustersIds(mainSelection);

	freeSelection(&mainSelection);
}
#endif // __CSLAVE_H