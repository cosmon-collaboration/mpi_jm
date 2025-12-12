#include <mpi.h>

// 
// Idea is to launch this with a first srun to reserve resources
// for spawns.
// 

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    /* Name this rank pool "all_nodes", which will be
     * used by MPI_Comm_spawn to identify it. */
    MPIX_Comm_rankpool(MPI_COMM_WORLD, "all_nodes", /* 1 hour timeout */ 3600);
    MPI_Finalize();
}
