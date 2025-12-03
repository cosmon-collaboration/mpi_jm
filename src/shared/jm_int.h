/*
 BSD 3-Clause License
 Copyright (c) 2025, The Regents of the University of California
 See Repository LICENSE file
 */
// 
// jm_int.h
// Shared by jm_master and jm_sched so messages are in sync
// 
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#define jm_setlinebuf(fp) setvbuf(fp, NULL, _IOLBF, BUFSIZ)

#define MPI_PROC_NULLx 0

#define LUMPNAMESIZE 128 // for transmission
#define JM_WELCOME_SIZE 16

#define JM_LOC_ENT_SIZE (MPI_MAX_PROCESSOR_NAME+1)
// initial data packet size from jm_master to jm_worker
#define JM_INTER_DATA_SIZE 16
#define JM_PASS_LOC 1

// define unique tags for debugging async communication
#define JM_CMD_TAG 23
#define JM_ARG_TAG 27
#define JM_STATUS_TAG 1000 /* + jobid */

#define JM_CMD_QUIT   1
#define JM_CMD_STATUS 2
#define JM_CMD_JOB    3
#define JM_CMD_NAP    4

// Encoding to pass data to "master" from jm_sched to
// describe jobs.
// include NUL in len
#define JM_BUF_JOBNAME "jobname:"
#define JM_BUF_JOBNAME_LEN 8
#define JM_BUF_PGM "pgm:"
#define JM_BUF_PGM_LEN 4
#define JM_BUF_STDOUT "stdout:"
#define JM_BUF_STDOUT_LEN 7
#define JM_BUF_DIR "dir:"
#define JM_BUF_DIR_LEN 4
#define JM_BUF_ARG "arg:"
#define JM_BUF_ARG_LEN 4
#define JM_BUF_ENV "env:"
#define JM_BUF_ENV_LEN 4
#define JM_BUF_RANK "rank:"
#define JM_BUF_RANK_LEN 5
#define JM_BUF_NTHREAD "nt:"
#define JM_BUF_NTHREAD_LEN 3
#define JM_BUF_STIMEOUT "sto:"
#define JM_BUF_STIMEOUT_LEN 4
#define JM_BUF_END "end:"
#define JM_BUF_END_LEN 4



/* suffix to apply to logfile stem */
#define JM_STARTED_SUF ".started"

// Common control for applicaiton spawn child disconnect which must
// happen in both master and the client library in a matching way.
// Only turn off for debugging mpi issues.  Otherwise, if the child
// crashes, the error will propagate.
#define JM_DISCONNECT_CHILD 1

#define streq(a, b) !strcmp(a, b)
#define strcaseeq(a, b) !strcasecmp(a, b)

// debugging
// test with #ifdef
// xtra allocation for certain new/malloc operations (debugging)
// #define JM_XALLOC 1024
#define JM_XALLOC 0

#define JM_MACH_PARMS_SIZE 8
#define JM_MACH_PARMS_VERSION 1001
