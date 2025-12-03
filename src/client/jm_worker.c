/*
 BSD 3-Clause License
 Copyright (c) 2025, The Regents of the University of California
 See Repository LICENSE file
 */

//
// jm_worker.c defines some utility functions needed
// by MPI jobs that are to be run under the job manager.
// See the example latworker.cc for usage.
//
// jobs compiled with jm_worker.c may be run separately.
// They will automatically detect that they have no parent.
//
// jm_parent_handshake(int *argc, char ***argv) 
//    To be called immedately after MPI_Init.   Does an
//    information exchange with the job manager and
//    disconnects for crash protection.
// jm_finish(int rc, char *message)
//    To be called after MPI_Finalize to record status
//
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "mpi.h" 
#include "../../config/config.h"
#include "jm_int.h"
#include "jm.h"

//
// Note:   Apparently, it is possible to make sched_getcpu with
// int cpu, numanode;
// int rc = syscall(SYS_getcpu, &cpu, &numanode, NULL)
// Todo: test on arm based Apple.

#ifdef USE_SCHED
#define _GNU_SOURCE 1
#define __USE_GNU 1
#include <sched.h>
#endif
#ifdef USEOMP
#include "omp.h"
#include <sys/syscall.h>
#endif

#ifdef __APPLE__
#include <mach/thread_policy.h>
#else
#include <sys/syscall.h>
int syscall(int number, ...); // why is this missing
#endif

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

// used to enable output from ranks other than 0
const char *jm_dump_rank_env = "JM_DUMP_RANK_ENV";

static char *jm_logfile = NULL;
static int jm_block_size;   // number of nodes in a block
static int jm_parent_rank;
static int jm_block_id;     // which block spawned worker
static int jm_spawn_rank;   // rank in spawned job, MPI_COMM_WORLD
static int jm_spawn_size;   // size of MPI_COMM_WORLD
static MPI_Comm jm_block_intercomm;  // intercomm with "block" parent
static char jm_loc[JM_LOC_ENT_SIZE]; // holds host name for this rank
//static char *jm_statusfile;

int jm_nthreadsperrank;    // number of OMP threads per rank

char *jm_mstr(const char *s) {
	size_t sz = strlen(s) + 1;
	char *buf = malloc(sz);
	if(!buf) {
		printf("jm_mstr: out of memory, request size %lu\n", (long unsigned)sz);
		exit(1);
	}
	strcpy(buf, s);
	return buf;
}

// visible to worker
int jm_active = 0; // 0 is FALSE.   Making this accessible to C clients.  Make type int
const char *jm_parentname = NULL;
const char *jm_jobname = NULL;

static void jm_set_logfile(char *passed_logfile) {
	//
	// TODO: Compute log file for this spawned process
	// We use  <working-dir>/<jobnamestem>.logtmp
	// When the process finishes successfully we rename *.logtmp to *.log
	// jm_master is looking for the disappearance of *.logtmp to indicate
	// a successful completion.
	// 
	if(jm_spawn_rank == 0 || getenv(jm_dump_rank_env)) {
		char *buf = NULL;
		char *cp;
		if(passed_logfile) {
			printf("Worker: received passed log file %s\n", passed_logfile);
			buf = (char *)malloc(strlen(passed_logfile) + 100);
			strcpy(buf, passed_logfile);
			cp = buf + strlen(buf);
			if(jm_spawn_rank == 0) {
				sprintf(cp, "tmp");
			} else {
				// other ranks get a number in the file.
				sprintf(cp, "tmp_R%d", jm_spawn_rank);
			}
		} else {
			// old way where we compute from jobname
			buf = (char *)malloc(strlen(jm_jobname) + 100);
			cp = strrchr(jm_jobname, '/');
			if(cp) {
				// we are going to rely on the working dir being set
				// so we just need the last component of the path.
				// The jobname path may differ above the "job" directory
				// anyway.
				strcpy(buf, cp + 1); // move last comp to front of buf.
			}
			cp = strrchr(buf, '.'); // before .yaml
			if(cp) {
				*cp = 0; // truncate at '.'
			}
			cp = buf + strlen(buf);
			if(jm_spawn_rank == 0) {
				strcpy(cp, ".logtmp");
			} else {
				// other ranks get a number in the file.
				sprintf(cp, "R%d.logtmp", jm_spawn_rank);
			}
		}
		printf("Worker R%d: logfile='%s'\n", jm_spawn_rank, buf);
		fflush(stdout);
		jm_logfile = buf; // save for later
	} else {
		jm_logfile = NULL;
	}
}

//
// Some applications will include marking a "started" file.
// We do this because some initialization steps in mpi and GPUs
// have been seen to hang on random nodes on summit.
// This way jm_master can have a timeout for the appearence of the
// .started file.
// 
void jm_mark_started() {
	if(!jm_active) return;
	// make sure all ranks reach this point.
	MPI_Barrier(MPI_COMM_WORLD);
	// if we are under mpi_jm, then jm_logfile will get set.
	if(jm_spawn_rank == 0 && jm_logfile) {
		size_t len = strlen(jm_logfile);
		char *buf = (char *)malloc(len + 10);
		if(!buf) {
			printf("Worker: jm_mark_started(): can't alloc small buffer!\n");
			exit(1); // will lead to teardown by jm_master, if MPI doesn't do it.
		}
		strcpy(buf, jm_logfile);
		char *cp = strrchr(buf, '.');
		if(!cp) {
			printf("Worker: logfile '%s' missing suffix\n", buf);
			exit(1); // will lead to teardown by jm_master, if MPI doesn't do it.
		}
		strcpy(cp, JM_STARTED_SUF);
		FILE *fp = fopen(buf, "w");
		if(!fp) {
			printf("Worker: Can't open .started file\n");
			exit(1);
		}
		free(buf);
		fclose(fp);
	}
}

//
// This routine must match jm_set_parent_info
// in the job manager (master) 
//
void jm_recv_parent_info() {
	int len;
	int jobnamesize;
	char *jobname;
	int data[JM_INTER_DATA_SIZE]; // needs to match size in jm_master.cc:send_parent_info
	int *slots; // holds cpu slots to bind ranks to
	int nranks, nranks_parent; // number of ranks in job
	char *logfile = NULL;
	int logfilesize;
	int pi;
	
	// broadcast from root (rank 0) of parent block to worker processes
	MPI_Bcast(data, JM_INTER_DATA_SIZE, MPI_INT, 0, jm_block_intercomm);
	jm_parent_rank = data[0];
	jm_block_size = data[1];
	jm_block_id = data[2];
	jobnamesize = data[3]; // including NUL
	jm_nthreadsperrank = data[4];
	nranks_parent = data[5];
	pi = data[6];
	logfilesize = data[7];
	if(jm_spawn_rank == 0)
		printf("Worker%d: Data %d:%d:%d:%d:%d:%d:%d:%d\n", jm_spawn_rank, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
	fflush(stdout);
	if(pi != 314159) {
		printf("Inconsistent jm_worker library with jm_master");
	}

	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	if(nranks != nranks_parent) {
		printf("Worker%d: disagreement over number of ranks with jm_master, %d vs %d\n", 
			jm_spawn_rank, nranks, nranks_parent);
		fflush(stdout);
	}
	slots = (int *)malloc(nranks * sizeof(int));
	MPI_Bcast(slots, nranks, MPI_INT, 0, jm_block_intercomm);

	jobname = malloc(jobnamesize+1 + JM_XALLOC);
	MPI_Bcast(jobname, jobnamesize, MPI_CHAR, 0, jm_block_intercomm); // from jm_master to job ranks
	jm_jobname = jobname;
	if(logfilesize > 0) {
		logfile = (char *)malloc(logfilesize);
		MPI_Bcast(logfile, logfilesize, MPI_CHAR, 0, jm_block_intercomm); // from jm_master to job ranks
	}

	jm_set_logfile(logfile);
	if(logfile) free(logfile);

	if(jm_spawn_rank == 0) {
		// Inform parent of logfile name
		int logfilesize = strlen(jm_logfile) + 1;
		MPI_Send(jm_logfile, logfilesize, MPI_CHAR, 0, 0, jm_block_intercomm);
	}

	// now send info back - processor and pid
	MPI_Get_processor_name(jm_loc, &len);
	jm_loc[len] = 0; // make sure


#ifdef __APPLE__
	// https://developer.apple.com/library/content/releasenotes/Performance/RN-AffinityAPI/index.html#//apple_ref/doc/uid/TP40006635-CH1-DontLinkElementID_2
	{
		printf("Worker: Mac OS/X: need way to lock rank %d to hardware thread %d\n", jm_spawn_rank, slots[jm_spawn_rank]);
		printf("There are nthreadsperrank=%d\n", jm_nthreadsperrank);
		fflush(stdout);
	}
#else
	{
		// References
		// https://stackoverflow.com/questions/1407786/how-to-set-cpu-affinity-of-a-particular-pthread
		// https://manpages.ubuntu.com/manpages/artful/man2/sched_getaffinity.2.html
		// GOMP_CPU_AFFINITY
		// OMP_PROC_BIND 

		// For now, assume that OMP threads get locked to a block of slots
		// The spec for affinity and GOMP is unclear about thread locking.
		// I tested this on lassen and it worked.    A possibly more portable way to do this
		// would be to use GOMP_CPU_AFFINITY on a per rank basis in the MPI_Comm_spawn_multiple call.
		// However, that would only work with the gcc implementation of OMP.  Decisions ...
		//
		int slot0 = slots[jm_spawn_rank]; // where thread 0 gets locked.
#ifdef USEOMP
#pragma omp parallel
		{ // launch all the available OMP threads
			int tid = omp_get_thread_num(); // 0..numthreads-1
			cpu_set_t cset;
			CPU_ZERO(&cset);
			CPU_SET(slot0+tid, &cset);  // make set with just one entry
			pid_t xid = syscall(SYS_gettid); // no gettid on sierra/lassen/summit
			sched_setaffinity(xid, sizeof(cpu_set_t), &cset);
		}
#else
		pid_t xid = getpid();
		cpu_set_t cset;
		CPU_ZERO(&cset);
		CPU_SET(slot0, &cset);
		sched_setaffinity(xid, sizeof(cpu_set_t), &cset);
#endif
	}
#endif

#if 0
	// Below print not needed, see jm_report_ranks
	int cpuid = jm_getcpu();
	// TODO:  Gather cpuid values to rank 0 of job and print from there.
	//        Will also need hosts
	  
	printf("Worker: B%dR%d: proc %s:%d job %s, tslot=%d\n", jm_block_id, jm_spawn_rank, jm_loc, cpuid, jm_jobname, slots[jm_spawn_rank]);
	fflush(stdout);
#endif
	  
	// Send processor names back to jm_master
	MPI_Gather(jm_loc, JM_LOC_ENT_SIZE, MPI_CHAR, (char *)0, 0, MPI_CHAR, 0, jm_block_intercomm);
	char *nbuf = malloc(32);
	sprintf(nbuf, "Block:%d", jm_block_id);
#ifdef JM_DELETE
	if(jm_parentname) free(jm_parentname);
#endif
	jm_parentname = nbuf;

	if(jm_spawn_rank == 0) {
		printf("Worker B%d: About to send pids to jm_master\n",  jm_block_id);
		fflush(stdout);
	}

	// now send back pid info so the job can be tracked
	// The parent needs to know what has happened to the kids
	int64_t pbuf[1];
	pbuf[0] = getpid();
	// Send process ids back to jm_master
	MPI_Gather(pbuf, 1, MPI_INT64_T, (char *)0, 0, MPI_INT64_T, 0, jm_block_intercomm);
	free((void *)slots);
	if(jm_spawn_rank == 0) {
		printf("Worker B%d: jm_recv_parent_info done\n",  jm_block_id);
		fflush(stdout);
	}
}

//
// Report layout of job in nodes/slots
// Report to go in head of log file for the specific job instead
// of the overall mpi_jm "lump#.log" files.
// This will allow us to check that the job is laid out properly on the resources
//
static void jm_report_ranks() {
	// First, collect info from ranks
	//    rank, host, pid, slot
	int nranks = jm_spawn_size;
	int rank = jm_spawn_rank;

	int64_t mypid = getpid();
	int64_t *pids = NULL;
	int *tslots = NULL;
	int mytslot;
	char *hostnames = NULL;
	if(rank == 0) {
		printf("jm_report_ranks: nranks=%d\n", nranks);
		fflush(stdout);
		// allocate receive buffers, only in rank 0
		pids = (int64_t *)malloc(nranks * sizeof(mypid));
		hostnames = (char *) malloc(nranks * JM_LOC_ENT_SIZE);
		tslots = (int *)malloc(nranks * sizeof(int));
	}
	if(rank == 0) {
		printf("Gathering pids\n");
		fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(&mypid, 1, MPI_INT64_T, pids, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
	if(rank == 0) {
		printf("Gathering hostnames\n");
		fflush(stdout);
	}
	MPI_Gather(jm_loc, JM_LOC_ENT_SIZE, MPI_CHAR, hostnames, JM_LOC_ENT_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
	// set tslot.  If we get the affinity mask
#if 1
#ifdef __APPLE__
	// https://stackoverflow.com/questions/33745364/sched-getcpu-equivalent-for-os-x
	mytslot = jm_getcpu();
#else
	mytslot = sched_getcpu();
#endif
#else
	{
		cpu_set_t cset;
		CPU_ZERO(&cset);
		sched_getaffinity(mypid, sizeof(cset), &cset);
		mytslot = -1; // not set
		for(int i = 0; i < CPU_SETSIZE; i++) {
			if(CPU_ISSET(i, &cset)) {
				mytslot = i;
				break; // only one option set at startup
			}
		}
	}
#endif
	MPI_Gather(&mytslot, 1, MPI_INT, tslots, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(rank == 0) {
		printf("Job layout by rank\n");
		printf("Rank    host     pid   slot\n");
		for(int r = 0; r < nranks; r++) {
			char *h = &hostnames[r*JM_LOC_ENT_SIZE];
			printf("%4d  %8s %6ld  %4d\n", r, h, (long)pids[r], tslots[r]);
		}
		fflush(stdout);
	}

	if(pids) free(pids);
	if(hostnames) free(hostnames);
	if(tslots) free(tslots);
}

//
// After talking with jm_master, redirect output to jm_logfile if set.
// It may be null in ranks > 0
//
// TODO:  Consider merging output from all ranks.   This may require
// all ranks to do a popen of an output server process where all
// ranks pass their data to the rank 0 server which actually writes the output.
// 
void jm_remap_stdout() {
	if(jm_logfile) {
		if(!freopen(jm_logfile, "w", stdout)) {
			printf("Worker: Unable to redirect stdout to '%s'\n", jm_logfile);
		} else {
			dup2(1, 2); // send stderr to same place
		}
		// log should be line buffered
		setvbuf(stdout, NULL, _IOLBF, BUFSIZ);
	} else {
		// don't write output from other than rank 0
		freopen("/dev/null", "w", stdout);
		dup2(1, 2); // send stderr to same place
	}
}

//
// Establish connection with parent
//
void jm_parent_handshake(int *argcp, char ***argvp) {
	char **argv = *argvp;
	int argc = *argcp;

	jm_active = 1; // 1 is TRUE
	jm_jobname = NULL;
	jm_parentname = jm_mstr("<self>");

	MPI_Comm_get_parent(&jm_block_intercomm); 
	if (jm_block_intercomm == MPI_COMM_NULL) {
		jm_jobname = jm_mstr("<job>");
		int cpuid = jm_getcpu();
		int len;
		MPI_Get_processor_name(jm_loc, &len);
		jm_loc[len] = 0; // make sure
		MPI_Comm_rank(MPI_COMM_WORLD, &jm_spawn_rank);
		if(getenv(jm_dump_rank_env)) {
			printf("Not under mpi_jm: worker starting on rank %d, on %s:%d\n", jm_spawn_rank, jm_loc, cpuid);
		} else {
			if(jm_spawn_rank == 0) {
				printf("Not under mpi_jm: worker starting on rank %d, on %s:%d\n", jm_spawn_rank, jm_loc, cpuid);
				printf("For other ranks set env %s\n", jm_dump_rank_env);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		return;
	}
	jm_active = 1; // 1 is TRUE

	// we were launched from the job manager, so perform exchange
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &jm_spawn_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &jm_spawn_size); 
	MPI_Comm_remote_size(jm_block_intercomm, &size); 
	jm_recv_parent_info();
	if (size != 1) {
		printf("Something's wrong with the parent\n"); 
		MPI_Abort(MPI_COMM_WORLD, 25);
	}
	if(jm_spawn_rank == 0) {
		printf("Work: B%dR%d:%s Attempting immediate disconnect\n", jm_block_id, jm_spawn_rank, jm_jobname);
	}
#ifdef JM_DISCONNECT_CHILD
	MPI_Comm_disconnect(&jm_block_intercomm);
#endif
	// only remap stdout/stderr after disconnect.
	jm_remap_stdout();
	if(jm_spawn_rank == 0) {
		printf("jm_worker: B%d:%s Disconnect complete\n", jm_block_id, jm_jobname);

		printf("jm_worker: Running %s under %s\n", argv[0], jm_parentname);
		for(int i = 1; i < argc; i++) {
			printf("jm_worker: argv[%d]: %s\n", i, argv[i]);
		}
		// system("env > workenv.log");
	}
	jm_report_ranks(); // needs all ranks
	MPI_Barrier(MPI_COMM_WORLD);
	if(jm_spawn_rank == 0) 
		printf("Continuing with application\n");
}

//
// produce status file to be read back by block root process
//
void jm_finish(int rc, const char *msg) {
	if(!jm_active) return;
	if(jm_spawn_rank == 0) {
		// printf("Worker: '%s' Job exiting with status %d\n",  jm_jobname, rc);
		// printf("Worker: '%s' Job exit message: %s\n", jm_jobname, msg);
		// We need to open a status file and write the info there.
		// So that the master block rank=0 process can read it back.

		// Automatic rename of log file from *.logtmp to *.log
		char *nbuf = jm_mstr(jm_logfile);
		// check for trailing tmp and remove
		int rc = -1;
		int len = strlen(nbuf);
		errno = 0;
		if(len > 3) {
			char *cp = nbuf + (len - 3);
			if(!strcmp(cp, "tmp")) {
				*cp = 0;
				rc = rename(jm_logfile, nbuf);
			}
		}
		fflush(stdout);
		if(rc < 0) {
			printf("Automatic rename of logfile *.logtmp -> *.log failed with errno=%d\n", errno);
			fflush(stdout);
		}
	}
}


// Some useful notes I found

/*
http://stackoverflow.com/questions/30511268/use-of-mpi-comm-self
Fortunately, MPI provides a caching mechanism that allows portable association of
arbitrary attributes to some MPI objects, namely communicators, windows and datatypes,
which is mainly useful when writing portable libraries. Each attribute has a set of
copy and delete callbacks that get called each time a certain event happens,
for example when an attribute gets copied as a result of the duplication of a communicator.
The standard does not give any guarantee on the order in which all MPI objects
get destroyed during MPI_Finalize() but it guarantees that MPI_COMM_SELF is the first
one to get destroyed. So attaching an attribute with a delete callback to 
MPI_COMM_SELF will trigger the callback right after the call to MPI_Finalize().
*/

#ifdef __APPLE__

#include <mach-o/dyld.h>	/* _NSGetExecutablePath */
#include <limits.h>		/* PATH_MAX */
#include <unistd.h>

void PDebugStop() {
	printf("In PDebugStop\n");
}

void PDebugMe() {
	char buf[32];
	char epath[PATH_MAX+1];
	uint32_t epathlen = PATH_MAX;
	_NSGetExecutablePath(epath, &epathlen);
	pid_t pid = getpid();
	sprintf(buf, ".debug%d", (int) pid);
	FILE *fp = fopen(buf, "w");
	if(!fp) {
		printf("Debugger launch file open failed: %s\n", buf);
		return;
	}
	fprintf(fp, "osascript -e 'tell app \"Terminal\"\n");
	fprintf(fp, "   do script \"gdb '%s' --pid %d -ex \\\"b PDebugStop\\\" \"\n", epath, (int)pid );
	fprintf(fp, "end tell'\n");
	fclose(fp);
	char sbuf[128];
	sprintf(sbuf, "sh %s &", buf);
	system(sbuf);
	sleep(2);
	PDebugStop();
}

#else
void PDebugMe() {
	printf("Don't known how to launch debugger on this platform\n");
	exit(1);
}


#endif


#ifdef USE_CPUID
// on Intel, use cpuid instruction to find logical processor
#include <cpuid.h>
#include <sys/types.h>
#include <sys/sysctl.h>

// http://stackoverflow.com/questions/33745364/sched-getcpu-equivalent-for-os-x
#define CPUID(INFO, LEAF, SUBLEAF) __cpuid_count(LEAF, SUBLEAF, INFO[0], INFO[1], INFO[2], INFO[3])

#define GETCPU(CPU) {                              \
        uint32_t CPUInfo[4];                           \
        CPUID(CPUInfo, 1, 0);                          \
        /* CPUInfo[1] is EBX, bits 24-31 are APIC ID */ \
        if ( (CPUInfo[3] & (1 << 9)) == 0) {           \
          CPU = -1;  /* no APIC on chip */             \
        } else {                                       \
          CPU = (unsigned)CPUInfo[1] >> 24;            \
        }                                              \
        if (CPU < 0) CPU = 0;                          \
      }

// Get logical CPU current execution is on
int jm_getcpu(void) {
	int cpuid;
	GETCPU(cpuid);
	return cpuid;
}
#else
//#include <sched.h>
extern int sched_getcpu(void);

int jm_getcpu(void) {
#ifdef USE_SCHED
	return sched_getcpu();
#else
	return -1;
#endif
}
#endif

// wrap stupid interface getcwd with one that
// increases buffer size until success
// don't use (NULL, 0) option because it will use malloc
char *jm_getcwd(void) {
	int sz = 256;
	char *dirbuf = malloc(sz);
	char *buf;
	while(1) {
		buf = getcwd(dirbuf, sz);
		if(buf) return buf;
		free(dirbuf);
		sz += 256;
		dirbuf = malloc(sz);
	}
}

void jm_freecwd(char *cwd) { if(cwd) free(cwd); }
