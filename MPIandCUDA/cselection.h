#ifndef __CSELECTION_H
#define __CSELECTION_H

#include <cuda.h>

typedef struct 
{
    float* items;
    int* clustersIds;
    
    float* d_items;
    int* d_clustersIds;

    int itemsCount;
    int paramsCount;
    cudaStream_t stream;

} cselection;

cselection createSelection(int itemsCount, int paramsCount)
{
    cselection s;

    s.paramsCount = paramsCount;
    s.itemsCount = itemsCount;

    s.items = (float*)malloc(itemsCount * paramsCount * sizeof(float));
    s.clustersIds = (int*)malloc(itemsCount * sizeof(int));

    return s;
}

void copySelectionToDevice(cselection* s)
{
    int itemsCount = s->itemsCount;
    int paramsCount = s->paramsCount;

    cudaMalloc((void**)&(s->d_items), itemsCount * paramsCount * sizeof(float));
    cudaMemcpy(s->d_items, s->items, itemsCount * paramsCount * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(s->d_clustersIds), itemsCount * sizeof(float));
    cudaMemcpy(s->d_items, s->items, itemsCount * sizeof(float), cudaMemcpyHostToDevice);
}

cselection* distributeSelection(cselection s, int processesCount)
{
    cselection* selections = (cselection*)malloc(processesCount * sizeof(cselection));

    int itemsPerProcess = s.itemsCount / processesCount;
    int itemsRest = s.itemsCount % processesCount;
    int distributedItems = 0;

    for(int i = 0; i < processesCount; i++)
    {
        selections[i].paramsCount = s.paramsCount;
        selections[i].itemsCount = itemsPerProcess;

        if(i == 0)
        {
            selections[i].itemsCount += itemsRest;
        }

        selections[i].items = s.items + distributedItems * s.paramsCount;
        selections[i].clustersIds = s.clustersIds + distributedItems;

        distributedItems += selections[i].itemsCount;
    }

    return selections;
}

void freeSelection(cselection* s)
{
    free(s->items);
    free(s->clustersIds);
}

void freeSelectionOnDevice(cselection* s)
{
    cudaFree(s->d_items);
    cudaFree(s->d_clustersIds);
}

#endif // __CSELECTION_H