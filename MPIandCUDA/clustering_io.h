#ifndef __CLUSTERING_IO_H
#define __CLUSTERING_IO_H

#include <mpi.h>

void readFile(MPI_File file, float* items, int offset, int size)
{
    MPI_File_set_view(file, offset * sizeof(float), MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_read(file, items, size, MPI_FLOAT , MPI_STATUS_IGNORE);
}

void writeFile(MPI_File file, float* items, int offset, int itemsCount, int paramsCount,
     int* clustersIds)
{
    MPI_File_set_view(file, offset, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

    for(int i = 0; i < itemsCount; i++)
    {
        MPI_File_write(file, &clustersIds[i], 1, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_write(file, items + i * paramsCount, paramsCount, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
}

#endif // __CLUSTERING_IO_H