#include <mpi.h>
#include "csystem.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    csystem dsystem;

    MPI_Comm_size(MPI_COMM_WORLD, &dsystem.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &dsystem.rank);

    systemInitialization(&dsystem, argc, argv);

    runClusteringSystem(dsystem);

    MPI_Finalize();
    return 0;
}

           