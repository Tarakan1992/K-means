#ifndef __CSYSTEM_H
#define __CSYSTEM_H


#define FILE_NAME_INDEX  1
#define ITEMS_COUNT_INDEX  2
#define PARAMS_COUNT_INDEX  3
#define CLUSTERS_COUNT_INDEX 4

typedef struct 
{
    int rank;
    int size;
    int paramsCount;
    int itemsCount;
    int clustersCount;
    char* filename;
} csystem;

void printFloatMassive(float* ms, int count)
{
    for(int i = 0; i < count; i++)
    {
        printf("%f ", ms[i]);
    }

    printf("\n");
}

void printIntMassive(int* ms, int count)
{
    for(int i = 0; i < count; i++)
    {
        printf("%d ", ms[i]);
    }

    printf("\n"); 
}

#include "cmaster.h"
#include "cslave.h"

void systemInitialization(csystem* s, int argc, char** argv)
{
    if(argc < 5)
    {
        if(s->rank == 0)
        {
            printf("Parameters must be {file} {number of items} {number of params} {number of clusters}\n");    
        }

        exit(-1);
    }

    s->itemsCount = atoi(argv[ITEMS_COUNT_INDEX]);
    s->paramsCount = atoi(argv[PARAMS_COUNT_INDEX]);
    s->clustersCount = atoi(argv[CLUSTERS_COUNT_INDEX]);
    s->filename = argv[FILE_NAME_INDEX];

    if(s->itemsCount < 1 || s->paramsCount < 1 || s->clustersCount < 1)
    {
        if(s->rank == 0)
        {
            printf("INVALID PARAMETERS");
        }
            
        exit(-1);
    }
}

void runClusteringSystem(csystem dsystem)
{
    if(dsystem.rank == 0)
    {
        runMasterProcess(dsystem);
    }
    else
    {
        runSlaveProcess(dsystem);
    }
}


#endif // __CSYSTEM_H