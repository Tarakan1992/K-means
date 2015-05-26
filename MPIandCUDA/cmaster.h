#ifndef __CMASTER_H
#define __CMASTER_H

#include <mpi.h>

#include "cselection.h"
#include "csystem.h"
#include "ciomodule.h"
#include "clustering.h"


void sendItemsCount(cselection* selections, int size)
{
    for(int destination = 0; destination < size; destination++)
    {
        MPI_Send((void*)&selections[destination].itemsCount, 1, 
            MPI_INT, destination + 1, 0, MPI_COMM_WORLD);
    }
}

void sendItems(cselection* selections, int size)
{
    for(int destination = 0; destination < size; destination++)
    {
         MPI_Send((void*)selections[destination].items, selections[destination].itemsCount * 
            selections[destination].paramsCount, MPI_FLOAT, destination + 1, 0, MPI_COMM_WORLD);
    }
}

void sendCenters(float* centers, int clustersCount, int paramsCount, int size)
{
    for(int destination = 0; destination < size; destination++)
    {
        MPI_Send((void*)centers, clustersCount * paramsCount, 
            MPI_FLOAT, destination + 1, 0, MPI_COMM_WORLD);
    }
}

void sendEOC(int* EOC, int size)
{
    for(int destination = 0; destination < size; destination++)
    {
        MPI_Send((void*)EOC, 1, MPI_INT, destination + 1, 0, MPI_COMM_WORLD);
    }
}

void reciveClustersIds(cselection* selections, int size)
{
    MPI_Status status;

    for(int source = 0; source < size; source++)
    {
        MPI_Recv((void*)selections[source].clustersIds, selections[source].itemsCount,
            MPI_INT, source + 1, 0, MPI_COMM_WORLD, &status);
    }
}

void reciveNewCenters(ccenters* cs, int size, int clustersCount, int paramsCount)
{
    MPI_Status status;

    for(int source = 0; source < size; source++)
    {
        MPI_Recv((void*)cs[source].newCenters, clustersCount * paramsCount, 
            MPI_FLOAT, source + 1, 0, MPI_COMM_WORLD, &status);

        MPI_Recv((void*)cs[source].itemsPerClusters, clustersCount, 
             MPI_INT, source + 1, 0, MPI_COMM_WORLD, &status);
    }
}

void runMasterClusteringCycle(cselection* selections, int size, ccenters c, int clustersCount, int paramsCount)
{
    int EOC;

   // int temp = 0;

    ccenters* cs = distributeCenters(c, size);

    do
    {
        sendCenters(c.centers, clustersCount, paramsCount, size);
        
        memset(c.newCenters, 0.0, clustersCount * paramsCount *sizeof(float));
        memset(c.itemsPerClusters, 0, clustersCount * sizeof(int));

        reciveNewCenters(cs, size, clustersCount, paramsCount);

        for(int i = 0; i < size; i++)
        {
            for(int j = 0; j < clustersCount; j++)
            {
                for(int k = 0; k < paramsCount; k++)
                {
                    c.newCenters[j * paramsCount + k] += cs[i].newCenters[j * paramsCount + k];
                }

                c.itemsPerClusters[j] += cs[i].itemsPerClusters[j];
            }
        }


        // for(int j = 0; j < clustersCount; j++)
        // {
        //     printf("%d items\n", c.itemsPerClusters[j]);
        //     printFloatMassive(c.newCenters + j * paramsCount, paramsCount);
        // }   
    

        for(int i = 0; i < clustersCount; i++)
        {
            for(int j = 0; j < paramsCount; j++)
            {
                c.newCenters[i * paramsCount + j] /= c.itemsPerClusters[i];
            }
        }

        // printf("Master recive centers\n");
        // for(int i = 0; i < clustersCount; i++)
        // {
        //     printFloatMassive(c.newCenters + i * paramsCount, paramsCount);
        // }

        //printf("\n\n");
        //printFloatMassive(c.newCenters, clustersCount * paramsCount);
            
        //if(temp == 1)
        //{
        //    EOC = 1;
        EOC = centersIsEqual(c.centers, c.newCenters, clustersCount, paramsCount);
        //}

        //temp++;
        
        sendEOC(&EOC, size);

        memcpy(c.centers, c.newCenters, clustersCount * paramsCount * sizeof(float));

        printf("|");

    }while(!EOC);
}

void runMasterProcess(csystem dsystem)
{
    int itemsCount = dsystem.itemsCount;
    int paramsCount = dsystem.paramsCount;
    int clustersCount = dsystem.clustersCount;
    int size = dsystem.size - 1;
    double start, end;

    start = MPI_Wtime();

    cselection mainSelection = createSelection(itemsCount, paramsCount);
    
    loadItemsFromFile(dsystem.filename, mainSelection.items, itemsCount, paramsCount);

    ccenters c = createCenters(clustersCount, paramsCount);

    initCentersOfClusters(mainSelection.items, c.centers, itemsCount, paramsCount, clustersCount);

    cselection* selections = distributeSelection(mainSelection, size);

    sendItemsCount(selections, size);
    sendItems(selections, size);

    runMasterClusteringCycle(selections, size, c, clustersCount, paramsCount);

    reciveClustersIds(selections, size);

    end = MPI_Wtime();

    printf("time: %.20lf\n", end - start);

    saveItemsInFile("output.txt", mainSelection.items, mainSelection.clustersIds, itemsCount, paramsCount);

    freeSelection(&mainSelection);
    freeCenters(&c);
}



#endif // __CMASTER_H