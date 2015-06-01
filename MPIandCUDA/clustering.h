#ifndef __CLUSTERING_H
#define __CLUSTERING_H

#include <stdlib.h>
#include <mpi.h>
#include <cuda.h>

#include "clustering_io.h"

#include "clustering_kernels.cu"

#define RAND_INIT 10
#define min(X, Y) (((X) < (Y)) ? (X) : (Y))


typedef struct 
{

    float* items;
    float* d_items;
    float* d_centers;
    float* newCenters;
    float* d_newCenters;

    int* d_itemsPerClusters;
    int* itemsPerClusters;
    int* clustersIds;
    int* d_clustersIds;


    int itemsCount;
    int offset;

    cudaStream_t stream;

    int gridSize;
    int blockSize;
    int sharedMemory;

} selection;


void initCentersOfClusters(float* centers, int clustersCount, int paramsCount)
{
    srand(RAND_INIT);

    for(int i = 0; i < clustersCount; i++)
    {
        for(int j = 0; j < paramsCount; j++)
        {
            centers[i * paramsCount + j] = (float)((rand() % 8000) * 0.01);
        }
    }    
}

int centersIsEqual(float* centers1, float* centers2, int clustersCount, int paramsCount)
{
    for(int i = 0; i < clustersCount * paramsCount; i++)
    {
        if(abs(centers1[i] - centers2[i]) > 0.0001) 
        {
            return 0;
        }
    }

    return 1;
}


int endOfClustering(float* centers, float* newCenters, int* itemsPerClusters,
    int clustersCount, int paramsCount, int size, int rank)
{
    int EOC;

    if(rank == 0)
    {
        float* tNewCenters = (float*)malloc(clustersCount * paramsCount * sizeof(float));
        int* tItemsPerClusters = (int*)malloc(clustersCount * sizeof(int));

        for(int source = 1; source < size; source++)
        {
            MPI_Recv((void*)tNewCenters, clustersCount * paramsCount, 
                MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Recv((void*)tItemsPerClusters, clustersCount, 
                 MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
            for(int i = 0; i < clustersCount; i++)
            {
                itemsPerClusters[i] += tItemsPerClusters[i];
                for(int j = 0; j < paramsCount; j++)
                {
                    newCenters[i * paramsCount + j] += tNewCenters[i * paramsCount + j];
                }
            }
        }

        for(int i = 0; i < clustersCount; i++)
        {
            for(int j = 0; j < paramsCount; j++)
            {
                if(itemsPerClusters[i] != 0)
                {
                    newCenters[i * paramsCount + j] /= itemsPerClusters[i];    
                }
                else
                {
                    newCenters[i * paramsCount + j] = centers[i * paramsCount + j];
                }
            }
        }

        EOC = centersIsEqual(centers, newCenters, clustersCount, paramsCount);

        memcpy(centers, newCenters, clustersCount * paramsCount * sizeof(float));

        for(int destination = 1; destination < size; destination++)
        {
            MPI_Send((void*)&EOC, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);

            if(!EOC)
            {
                MPI_Send((void*)centers, clustersCount * paramsCount, MPI_FLOAT, 
                    destination, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        MPI_Send((void*)newCenters, clustersCount * paramsCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send((void*)itemsPerClusters, clustersCount, MPI_INT, 0, 0, MPI_COMM_WORLD);

        MPI_Recv((void*)&EOC, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if(!EOC)
        {
            MPI_Recv((void*)centers, clustersCount * paramsCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    return EOC;
}


void kMeans(char* filename, int itemsCount, int offset, int clustersCount, int paramsCount, int size, int rank)
{
            MPI_File input;
            MPI_File output;
            float* items;
            float* centers;
            float* newCenters;
            int* itemsPerClusters;
            int* clustersIds;
            int ierr;
            int EOC;
            int deviceCount;
            cudaDeviceProp dp;
            
    //allocation
    items = (float*)malloc(itemsCount * paramsCount * sizeof(float));
    centers = (float*)malloc(clustersCount * paramsCount * sizeof(float));
    clustersIds = (int*)malloc(itemsCount * sizeof(int));
    newCenters = (float*)malloc(clustersCount * paramsCount * sizeof(float));
    itemsPerClusters = (int*)malloc(clustersCount * sizeof(float));


    //get cuda device count
    cudaGetDeviceCount(&deviceCount);

    //selection for each device
    selection* selections = (selection*)malloc(deviceCount * sizeof(selection));

    //init selection for each device
    for(int i = 0; i < deviceCount; i++)
    {
        cudaSetDevice(i);
        //create stream
        cudaStreamCreate(&(selections[i].stream));

        //items for device
        selections[i].itemsCount = (itemsCount + deviceCount - 1) / deviceCount;

        //file offset for device
        selections[i].offset = i * selections[i].itemsCount;

        //correction for last device
        if(deviceCount != 1 && i == deviceCount - 1)
        {
            selections[i].itemsCount -= 1;
        }


        //set pointer
        selections[i].items = items + selections[i].offset * paramsCount;
        selections[i].clustersIds = clustersIds + selections[i].offset;

        //device allocation
        cudaMalloc((void**)&selections[i].d_items, selections[i].itemsCount * paramsCount * sizeof(float));
        cudaMalloc((void**)&selections[i].d_clustersIds, selections[i].itemsCount * sizeof(int));
        cudaMalloc((void**)&selections[i].d_centers, clustersCount * paramsCount * sizeof(float));
        cudaMalloc((void**)&selections[i].d_newCenters, clustersCount * paramsCount * sizeof(float));
        cudaMalloc((void**)&selections[i].d_itemsPerClusters, clustersCount * sizeof(int));

        selections[i].newCenters = (float*)malloc(clustersCount * paramsCount * sizeof(float));
        selections[i].itemsPerClusters = (int*)malloc(clustersCount * sizeof(int));

        cudaGetDeviceProperties(&dp, i);

        //device prop
        selections[i].blockSize = min(selections[i].itemsCount,  dp.maxThreadsPerBlock);
        selections[i].gridSize = (selections[i].itemsCount + selections[i].blockSize - 1) / selections[i].blockSize;
        selections[i].sharedMemory = dp.sharedMemPerBlock;
    }


    //open file
    ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input);
    if (ierr) 
    {
        if (rank == 0) 
        {
            printf("Couldn't open file %s\n", filename);
        }

        MPI_Finalize();
        exit(3);
    }


    //read items from file for each device
    for(int i = 0; i < deviceCount; i++)
    {
        cudaSetDevice(i);

        readFile(input, selections[i].items, (selections[i].offset + offset) * paramsCount, 
            selections[i].itemsCount * paramsCount);

        cudaMemcpyAsync(selections[i].d_items, selections[i].items,
            selections[i].itemsCount * paramsCount * sizeof(float),
            cudaMemcpyHostToDevice, selections[i].stream);
    }

    
    MPI_File_close(&input);

    initCentersOfClusters(centers, clustersCount, paramsCount);

    do
    {
        for(int i = 0; i < deviceCount; i++)
        {

            cudaSetDevice(i);

            cudaMemcpyAsync(selections[i].d_centers, centers, clustersCount * 
                paramsCount * sizeof(float), cudaMemcpyHostToDevice, selections[i].stream);

            clusreringKernel<<<selections[i].gridSize, selections[i].blockSize,
                selections[i].sharedMemory, selections[i].stream>>>(
                selections[i].d_items,
                selections[i].d_centers,
                selections[i].d_clustersIds,
                selections[i].itemsCount,
                clustersCount,
                paramsCount);


            cudaMemsetAsync(selections[i].d_newCenters, 0.0,
                clustersCount * paramsCount * sizeof(float), selections[i].stream);

            cudaMemsetAsync(selections[i].d_itemsPerClusters, 0,
                clustersCount * sizeof(int), selections[i].stream);

            newCentersKernel<<<clustersCount, paramsCount, selections[i].sharedMemory,selections[i].stream>>>(
                selections[i].d_items,
                selections[i].d_clustersIds,
                selections[i].d_newCenters,
                selections[i].d_itemsPerClusters,
                selections[i].itemsCount,
                clustersCount,
                paramsCount);

            cudaMemcpyAsync(selections[i].newCenters, selections[i].d_newCenters,
                clustersCount * paramsCount * sizeof(float), cudaMemcpyDeviceToHost, selections[i].stream);

            cudaMemcpyAsync(selections[i].itemsPerClusters, selections[i].d_itemsPerClusters,
               clustersCount  * sizeof(int), cudaMemcpyDeviceToHost, selections[i].stream);

        }

        for (int i = 0; i < deviceCount; i++)
        {
            cudaSetDevice(i);
            cudaStreamSynchronize(selections[i].stream);
        }

        memset(newCenters, 0.0, clustersCount * paramsCount * sizeof(float));
        memset(itemsPerClusters, 0, clustersCount * sizeof(int));

        for(int i = 0; i < deviceCount; i++)
        {   
            for(int j = 0; j < clustersCount; j++)
            {
                itemsPerClusters[j] += selections[i].itemsPerClusters[j];

                for(int k = 0; k < paramsCount; k++)
                {
                    newCenters[j * paramsCount + k] += selections[i].newCenters[j * paramsCount + k];
                }
            }
        }

        EOC = endOfClustering(centers, newCenters, itemsPerClusters, 
            clustersCount, paramsCount, size, rank);

    }while(!EOC);


    for(int i = 0; i < deviceCount; i++)
    {
        cudaSetDevice(i);
        cudaMemcpyAsync(selections[i].clustersIds, selections[i].d_clustersIds,
                selections[i].itemsCount * sizeof(int), cudaMemcpyDeviceToHost, selections[i].stream);
    }

    // if(rank == 0)
    // {
    //     for(int i = 0; i < itemsCount; i++)
    //     {
    //         for(int j = 0; j < paramsCount; j++)
    //         {
    //             printf("%f ", items[i * paramsCount + j]);
    //         }
    //         printf("\n");
    //     }

    //     printf("\n");

    //     for(int i = 0; i < itemsCount; i++)
    //     {
    //         printf("%d ", clustersIds[i]);
    //     }

    //     printf("\ncenters: \n");

    //     for(int i = 0; i < clustersCount; i++)
    //     {
    //         for(int j = 0; j < paramsCount; j++)
    //         {
    //             printf("%f ", centers[i * paramsCount + j]);
    //         }

    //         printf("\n");
    //     }
    // } 


    //open file
    ierr = MPI_File_open(MPI_COMM_WORLD, "output.bin", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &output);
    if (ierr) 
    {
        if (rank == 0) 
        {
            printf("Couldn't open file %s\n", "output.bin");
        }

        MPI_Finalize();
        exit(3);
    }

    //printf("rank: %d offset: %d itemsCount: %d\n", rank, offset * (sizeof(int) + paramsCount * sizeof(float)), itemsCount);

    writeFile(output, items, offset * (sizeof(int) + paramsCount * sizeof(float)), 
        itemsCount, paramsCount, clustersIds);

    for(int i = 0; i < deviceCount; i++)
    {
        cudaSetDevice(i);
        cudaFree(selections[i].d_items);
        cudaFree(selections[i].d_clustersIds);
        cudaFree(selections[i].d_centers);
        cudaFree(selections[i].d_newCenters);
        cudaFree(selections[i].d_itemsPerClusters);
    }

    MPI_File_close(&output);

}

#endif // __CLUSTERING_H