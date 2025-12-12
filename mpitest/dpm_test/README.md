# Introduction  

Mpi\_Jm uses a global name service, part of the DPM functionality, to establish connections
between the scheduler and block head nodes.   It also uses a global name service to allow
launching of groups of blocks independently to speed up startup for large numbers of nodes
as well as surviving bad nodes.

# go\_mvapichplus.sh

This script starts the hydra\_nameserver to allow connections between multiple mpiexec.hydra runs.

# go\_openmpi.sh

This script is set up for the 4.x.x openmpi series.  The 5.x.x series is currently broken and spawn doesn't work.

The script starts the ompi_nameserver, which provides the global name service.

```sh
ompi-server --no-daemonize -r $namefile &
```
