/*
 BSD 3-Clause License
 Copyright (c) 2025, The Regents of the University of California
 See Repository LICENSE file
 */

#define TRUE 1
#define FALSE 0

#define MAXBLOCKNAME 128

#ifdef OPENMPI
// Openmpi 5.0.7+
#define USE_MPI_OPEN_PORT 0
#define PORTFILE "mpiportfile.txt"
#else
// MVAPICH2-2.3.7
#define USE_MPI_OPEN_PORT 0
#define LINK_PORT "link_port"
#endif
#define SERVICE_NAME "jm"
