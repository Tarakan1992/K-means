#ifndef __CLUSTERING_H
#define __CLUSTERING_H

#include <stdlib.h>

typedef struct 
{
    float* centers;
    int clustersCount;
    int paramsCount;

    float* newCenters;
    int* itemsPerClusters;

    float* d_newCenters;
    int* d_itemsPerClusters;

} ccenters;

ccenters createCenters(int clustersCount, int paramsCount)
{
    ccenters c;
    c.clustersCount = clustersCount;
    c.paramsCount = paramsCount;

    c.centers = (float*)malloc(clustersCount * paramsCount * sizeof(float));
    c.newCenters = (float*)malloc(clustersCount * paramsCount * sizeof(float));
    c.itemsPerClusters = (int*)malloc(clustersCount * sizeof(int));

    return c;
}

void freeCenters(ccenters* c)
{
    free(c->centers);
    free(c->newCenters);
    free(c->itemsPerClusters);
}

void freeCentersOnDevice(ccenters* c)
{
    cudaFree(c->d_newCenters);
    cudaFree(c->d_itemsPerClusters);
}

void copyCentersToDevice(ccenters* c)
{
    int clustersCount = c->clustersCount;
    int paramsCount = c->paramsCount;

    cudaMalloc((void**)&(c->d_newCenters), clustersCount * paramsCount * sizeof(float));
    cudaMemcpy(c->d_newCenters, c->newCenters, clustersCount * paramsCount * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(c->d_itemsPerClusters), clustersCount * sizeof(int));
    cudaMemcpy(c->d_itemsPerClusters, c->itemsPerClusters, clustersCount * sizeof(int), cudaMemcpyHostToDevice);
}

ccenters* distributeCenters(ccenters c, int count)
{
    int clustersCount = c.clustersCount;
    int paramsCount = c.paramsCount;

    ccenters* cs = (ccenters*)malloc(count * sizeof(ccenters));

    for(int i = 0; i < count; i++)
    {
        cs[i].clustersCount = clustersCount;
        cs[i].paramsCount = paramsCount;
        cs[i].newCenters = (float*)malloc(clustersCount * paramsCount * sizeof(float));
        cs[i].itemsPerClusters = (int*)malloc(clustersCount * sizeof(int));
    }

    return cs;
}

void initCentersOfClusters(float* items, float* centers, int itemsCount, int paramsCount, int clustersCount)
{
    srand(time(NULL));

    int clustersIndexes[clustersCount];

    int isNew;

    int r;

    for(int i = 0; i < clustersCount; i++)
    {
        do
        {
            isNew = 1;
            r = rand() % itemsCount;
            
            for(int j = 0; j < i; j++)
            {
                if(clustersIndexes[j] == r)
                {
                    isNew = 0;
                    break;
                }
            }
        }
        while(!isNew);

        clustersIndexes[i] = r;

        for(int k = 0; k < paramsCount; k++)
        {
            centers[i * paramsCount + k] = items[r * paramsCount + k];
        }        
    }
}

int centersIsEqual(float* centers1, float* centers2, int clustersCount, int paramsCount)
{
    for(int i = 0; i < clustersCount * paramsCount; i++)
    {
        if(abs(centers1[i] - centers2[i]) > 0.0001) 
        //if(centers1[i] != centers2[i]) 
        {
            return 0;
        }
    }

    return 1;
}

#endif // __CLUSTERING_H