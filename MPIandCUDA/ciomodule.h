#ifndef __CIOMODULE_H
#define __CIOMODULE_H

#include <stdlib.h>

void loadItemsFromFile(char* filename, float* items, int itemsCount, int paramsCount)
{
    FILE* file = fopen(filename, "r");

    for(int i = 0; i < itemsCount; i++)
    {
        for(int j = 0; j < paramsCount; j++)
        {
            int error = fscanf(file, "%f", &items[i * paramsCount + j]);
            
            if(error == -1)
            {
                printf("Parameters error!");
                exit(0);
            }
        }
    }

    fclose(file);
}

void saveItemsInFile(char* filename, float* items, int* clustersIds, int itemsCount, int paramsCount)
{
    FILE* file = fopen(filename, "w");

    for(int i = 0; i < itemsCount; i++)
    {
        fprintf(file, "%d ", clustersIds[i]);

        for(int j = 0; j < paramsCount; j++)
        {
            fprintf(file, "%f ", items[i * paramsCount + j]);
        }

        fprintf(file,"\n");
    }

    fclose(file);
}

#endif // __CIOMODULE_H