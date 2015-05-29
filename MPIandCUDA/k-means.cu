#include <mpi.h>
#include <cuda.h>
#include "clustering.h"
#include "clustering_io.h"

#define FILE_NAME_INDEX  1
#define ITEMS_COUNT_INDEX  2
#define PARAMS_COUNT_INDEX  3
#define CLUSTERS_COUNT_INDEX 4


int main(int argc, char** argv)
{
	int rank, size;
	int itemsCount, paramsCount, clustersCount;
	int itemsPerProc, offset;
	int ierr;
	char* filename;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //read cl parameters
    if(argc < 5)
    {
        if(rank == 0)
        {
            printf("Parameters must be {file} {number of items} {number of params} {number of clusters}\n");    
        }

    	MPI_Finalize();
        exit(1);
    }

   	itemsCount = atoi(argv[ITEMS_COUNT_INDEX]);
    paramsCount = atoi(argv[PARAMS_COUNT_INDEX]);
    clustersCount = atoi(argv[CLUSTERS_COUNT_INDEX]);
    filename = argv[FILE_NAME_INDEX];

    if(itemsCount < 1 || paramsCount < 1 || clustersCount < 1)
    {
        if(rank == 0)
        {
            printf("INVALID PARAMETERS");
        }
    	MPI_Finalize();
        exit(2);
    }

    //items per mpi-process
    itemsPerProc = (itemsCount + size - 1) / size;

    //calc file offset
    offset = itemsPerProc * rank;

    //last process correction
    if(rank == size - 1)
    {
        itemsPerProc = itemsPerProc - 1;
    }


    kMeans(filename, itemsPerProc, offset, clustersCount, paramsCount, size, rank);


    MPI_Finalize();
    return 0;
}

           