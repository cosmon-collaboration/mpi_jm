//
// jm_master is launched 1 per node to monitor the node.
// The nodes are then organized into blocks (size specified in config file) that
// will be used to launch jobs.    Block members are chosen to have local interconnect.
//
// Each block will have a subcommunicator so the jm_masters can watch the pids of jobs
// running in the block.   The block rank 0 process will communicate with jm_sched via
// non blocking MPI calls to receive jobs to run and report status back.
//
// The block subcommunicator is used for spawning jobs.    MPI_Comm_spawn has enough
// control to pick the nodes.   It looks like it can also pick slots.   When jobs are
// sent from the scheduler they come with a complete assignment of node/slots per process
// for hybrid MPI/OMP jobs.
//
// TODO:
//    Use topology files from Gustav for block node selection
//
 

// info about binding to cores  
// http://stackoverflow.com/questions/19946796/can-we-get-core-id-on-which-the-process-is-running-in-mpi

//
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <vector>
#include <string>
#include "mpi.h" 
#include "../../config/config.h"
#include "jm_int.h"
using std::vector;

#if 0
void errsetline(const char *fname, int line);
void PMemVerifyFunc(const char *filename, int line);
void PNewInit();
#define SETLINE() errsetline(__FILE__, __LINE__)
#define MEMVERIFY() PMemVerifyFunc(__FILE__, __LINE__)

#else
#define SETLINE()  /* */
#define MEMVERIFY()  /* */
#define PNewInit() /* */
#endif

extern char *jm_mstr(const char *s);
extern char *jm_block_procname(int id);
extern void jm_unpackslotenv();

int jm_verbose = 1;
char *jm_sched_path = nullptr;
char *jm_spawnwrap_path = nullptr;

//
// The values here are overwritten from data delivered by the scheduler, which reads
// the config file.
//
int jm_block_size = -1;        // number of nodes grouped together for running jobs (-1 not set yet)
int jm_node_maxwidth = 2;      // max number of threads/processes we can run on a "node"
int jm_node_proccnt = 2;       // set to value used for job.

int jm_world_size;			   // top level number of "ranks" or starting number of MPI processes
int jm_world_rank;             // world rank of current process
int jm_block_id;               // world rank / jm_block_size for now.
int jm_block_rank;			   // rank position in block (parent)
int jm_block_count;            // number of blocks
MPI_Comm  jm_block_comm;       // comm between node owners in a block
MPI_Comm  jm_block_intercomm = MPI_COMM_NULL;  // intercomm between node owners and spawned job

MPI_Comm  jm_sched_intercomm = MPI_COMM_NULL;  // intercomm between world and scheduler.  Used for point to point comm

char *jm_block_locbuf;         // processor names of block processes (parents) stuck together
							   // in jm_block_size  blocks of JM_LOC_ENT_SIZE characters
char jm_bhostfile[1024];       // hosts we are allowed to launch on

char *jm_slotenvbuf = nullptr;

char jm_this_procname[JM_LOC_ENT_SIZE]; // Titan: nid03704

// used to hold slot level environment
typedef struct _jm_slotenv {
	int slot;
	char *name, *value;
	struct _jm_slotenv *next;
} jm_slotenv;

jm_slotenv **jm_slotenvlist = nullptr;
int jm_slotenv_xtra;  // extra buffer size for slotenv in env
static char jm_logbuf[4096];
//
// Fatal Error reporting
//
static void err(const char *msg, ...) {
	char *buf = jm_logbuf;
	va_list args;
	va_start(args, msg);
	vsprintf(buf, msg, args);
	va_end(args);
	printf("jm_master: %s\n", buf);
	MPI_Abort(MPI_COMM_WORLD, 17);
}

static void jm_log(const char *fmt, ...) {
	if(1 > jm_verbose) return;
	char *buf = jm_logbuf;
	va_list args;
	va_start(args, fmt);
	vsprintf(buf, fmt, args);
	va_end(args);
	printf("Master B%dR%d: %s", jm_block_id, jm_block_rank, buf);
}

static void jm_log(int lvl, const char *fmt, ...) {
	if(lvl > jm_verbose) return;
	char *buf = jm_logbuf;
	va_list args;
	va_start(args, fmt);
	vsprintf(buf, fmt, args);
	va_end(args);
	printf("Master B%dR%d: %s", jm_block_id, jm_block_rank, buf);
}

#define SPAWNWRAP
#define SPAWNWRAPPGM jm_spawnwrap_path

#ifdef SPAWNWRAP
// we don't actually quote the strings, this isn't going
// through a shell.
static char *jm_qstr(char *s) {
	char *buf = new char [strlen(s) + 4];
	strcpy(buf, s);
	return buf;
}
#endif

// make allocated copy of string s
char *jm_mstr(const char *s) {
	SETLINE();
	char *buf = new char[strlen(s) + 1];
	strcpy(buf,s);
	return buf;
}

// Get the current working directory
// return value is allocated with new.
char *jm_cc_getcwd() {
	int len = 1023;
	SETLINE();
	char *buf = new char[len+1];
	while(1){
		if(getcwd(buf, len)) return(buf);
		len += 256;
		delete[] buf;
		SETLINE();
		buf = new char[len + 1];
	}
}

// data structure for tracking the owning block rank and pid of
// each process in a job.
typedef struct {
	int brank;   // assigned owner of pid
	pid_t pid;   // pid on node associated with brank.  pid_t is a signed type
	int sig;     // sig to kill with, use SIGTERM first, then SIGKILL if it doesn't listen
	int rc;      // return code from process
} jm_spawn_pid;

class jm_spjob; // class for tracking running jobs.

jm_spjob *jm_spjoblist = NULL; // jobs running in block.

class jm_spjob {
public:
	int jobid;
	int proccnt, lastproccnt;
	char *jobname;
	int nthreadsperrank; // OMP thread count per rank to send to child for thread locking
	jm_spawn_pid *pids;
	int completionstatus;
	char *work_locbuf;
	char *wdir;
	char *logfile;
	char *startfile;  // file touched by app to indicate that is is up and running
	int start_time;    // time(NULL) at launch
	int start_timeout; // timeout in seconds
	int   start_fail;
	jm_spjob *next; // next job in list of running jobs.
	// constructor - reset fields
	jm_spjob(int ajobid, int aproccnt, int anthreadsperrank, const char *awdir) {
		jobid = ajobid;
		startfile = nullptr;
		start_time = -1;    // not set yet.  Set when job launch completes
		start_timeout = -1; // jobs that fail to mark <jobstem>.started get aborted after this much time
							  // can be set from python interface for job
		start_fail = false;   // has not failed to start yet.
		logfile = nullptr;
		SETLINE();
		jobname = new char [32];
		wdir = awdir ? jm_mstr(awdir) : nullptr;
		sprintf(jobname, "job%d", jobid);
		work_locbuf = NULL;
		nthreadsperrank = anthreadsperrank;
		completionstatus = 0;
		proccnt = aproccnt;
		lastproccnt = proccnt;
		SETLINE();
		pids = new jm_spawn_pid[proccnt];
		for(int i = proccnt; --i >= 0;) {
			pids[i].brank = -1;
			pids[i].pid = -1;
			pids[i].sig = SIGTERM;
			pids[i].rc = -1; // not set yet 
		}
		// link into job list
		next = jm_spjoblist;
		jm_spjoblist = this;
	}
	// remove job from job list
	void unlink() {
		jm_spjob **spjobp;
		spjobp = &jm_spjoblist;
		while(*spjobp != this) spjobp = &(*spjobp)->next;
		*spjobp = this->next;
	}
	// destructor
	~jm_spjob() {
		delete[] pids;
		delete[] jobname;
		delete[] work_locbuf;
		delete[] logfile;
		delete[] startfile;
		delete[] wdir;
		unlink();
	}
	void setjobname(const char *name) {
		delete[] jobname;
		jobname = jm_mstr(name);
	}
	void setlogfile(const char *name) {
		delete[] logfile;
		logfile = jm_mstr(name);
	}
	void finished(int rc) {
		MPI_Request statrequest;
		completionstatus = rc;
		if(jm_block_rank == 0) {
			jm_log("Job %s(%d) has finished with status %d\n", jobname, jobid, rc);
			MPI_Isend(&completionstatus, 1, MPI_INT, 0, 
						JM_STATUS_TAG+jobid, // each job gets unique tag
						jm_sched_intercomm, &statrequest);
			// we can just wait here, but sched process end may be
			// receiving multiple such messages in any order
			// It will use MPI_Test
			
			// jm_log("Waiting for job %s(%d) completion message delivery\n", jobname, jobid);
			MPI_Wait(&statrequest, MPI_STATUS_IGNORE);
			jm_log("Completion message, rc=%d, delivered to scheduler for job %s(%d)\n", rc, jobname, jobid);
		}
		MPI_Barrier(jm_block_comm);
	}
	void started() {
		MPI_Request statrequest;
		completionstatus = -1; // used to indicate starting of job
		if(jm_block_rank == 0) {
			// jm_log("job %s(%d) has been started\n", jobname, jobid);
			MPI_Isend(&completionstatus, 1, MPI_INT, 0, 
						JM_STATUS_TAG+jobid, // each job gets unique tag
						jm_sched_intercomm, &statrequest);
			// we can just wait here, but sched process end may be
			// receiving multiple such messages in any order
			// It will use MPI_Test
			
			// jm_log("Waiting for job %s(%d) start message\n", jobname, jobid);
			MPI_Wait(&statrequest, MPI_STATUS_IGNORE);
			jm_log("Start msg to scheduler for job %s(%d) delivered\n", jobname, jobid);
		}
		MPI_Barrier(jm_block_comm);
	}
	// get processor name for spid
	const char *spawn_procname(int spid) {
		if(spid < 0 || spid >= proccnt) err("spid %d out of range", spid);
		return work_locbuf + spid * JM_LOC_ENT_SIZE;
	}
	// get log file from rank 0 of spawned job
	// Also construct .started file from logfile
	void get_logfile() {
		if(jm_block_rank == 0) {
			int logfilesize;
			MPI_Status status;
			MPI_Probe(0, 0, jm_block_intercomm, &status);
			MPI_Get_count(&status, MPI_CHAR, &logfilesize);
			char *logtail = new char[logfilesize];
			MPI_Recv(logtail, logfilesize, MPI_CHAR, 0, 0, jm_block_intercomm, &status);
			if(wdir) {
				logfile = new char[strlen(wdir) + logfilesize + 2];
				sprintf(logfile, "%s/%s",  wdir, logtail);
				delete[] logtail;
			} else {
				logfile = logtail;
			}
			jm_log("Received log file '%s'\n", logfile);
			startfile = new char[strlen(logfile) + 10];
			strcpy(startfile, logfile); // creation indicates startup success
			char *cp = strrchr(startfile, '.');
			strcpy(cp, JM_STARTED_SUF);
		} else {
			logfile = nullptr; // other ranks don't bother.
			startfile = nullptr; // other ranks don't bother.
		}
	}
	void get_spawn_pids() {
		int i;
		// Now retrieve the pids of the spawned processes
		SETLINE();
		int64_t *pidbuf = new int64_t[proccnt];
		// the recvcnt is the number of entries per single source not the total number of entries.
		if(jm_block_rank == 0) {
			// gather pids from spawned job ranks to block rank 0
			MPI_Gather((char *)0, 0, MPI_INT64_T, pidbuf, 1, MPI_INT64_T, MPI_ROOT, jm_block_intercomm);
			// jm_log("broadcasting pids for spawned processes\n");
		}
		// send to all block ranks so they can monitor pids
		MPI_Bcast(pidbuf, proccnt, MPI_INT64_T, 0, jm_block_comm);
		// In each block rank compute which block rank owns each spawned pid
		// When testing we may have two block ranks on the same node.  Give ownership to the
		// first one.
		for(i = 0; i < proccnt; i++) { // for each spawned process
			int bid;
			const char *sprocname = spawn_procname(i); // get processor name it landed on
			pids[i].brank = -1;
			pids[i].pid = pidbuf[i];
			pids[i].sig = SIGTERM;  // if we need to stop it, first signal to use
			for(bid = jm_block_size; --bid >= 0;) { // match against processors jm_masters are on
				if(!strcmp(sprocname, jm_block_procname(bid))) {
					break;
				}
			}
			if(bid < 0) err("Unable to find parent for spawned processe in block");
			pids[i].brank = bid;
		}
		delete [] pidbuf;
	}

	// Use jm_block_intercomm to set parent info in spawned job
	// For now, just the rank of the root process in each block
	// Matching routine required in worker
	//
	// Note:  jm_block_intercomm was formed between MPI_COMM_SELF for rank 0 jm_master
	// and the ranks of the spawned job.   That is why communications with jm_block_intercomm
	// are restricted to jm_block_rank == 0.
	void send_parent_info(int *slots) {
		int data[JM_INTER_DATA_SIZE]; // initial data packet, matches jm_master
		memset(data, 0, JM_INTER_DATA_SIZE*sizeof(int));
		data[0] = jm_world_rank;
		data[1] = jm_block_size;
		data[2] = jm_block_id;
		data[3] = strlen(jobname) + 1;
		data[4] = nthreadsperrank;
		data[5] = proccnt; // number of ranks in spawn
		data[6] = 314159;
		// buffer size to receive log file name, if one is specified.
		data[7] = logfile ? strlen(logfile) + 1 : -1; // set logfile
		// TODO: also send file name for status
		if(jm_block_rank == 0) {
			// jm_log("Send parent header for %s(%d)\n", jobname, jobid);
			MPI_Bcast(data, JM_INTER_DATA_SIZE, MPI_INT, MPI_ROOT, jm_block_intercomm);

			// Send physical cpu slot to bind ranks to
			MPI_Bcast(slots, proccnt, MPI_INT, MPI_ROOT, jm_block_intercomm);
			// send name of job to worker
			MPI_Bcast(jobname, data[3], MPI_CHAR, MPI_ROOT, jm_block_intercomm);
			// jm_log("retrieving node names for spawned processes\n");
			if(logfile) {
				printf("Sending log file name %s\n", logfile);
				MPI_Bcast(logfile, data[7], MPI_CHAR, MPI_ROOT, jm_block_intercomm);
			}
		}
		MPI_Barrier(jm_block_comm);
		get_logfile();
		int locbufsize = proccnt * JM_LOC_ENT_SIZE;
		SETLINE();
		work_locbuf = new char [locbufsize];
		if(jm_block_rank == 0) {
			// Gather data from the spawned processes.
			// Root side gets the data.
			MPI_Gather((char *)0, 0, MPI_CHAR, work_locbuf, JM_LOC_ENT_SIZE, MPI_CHAR, MPI_ROOT, jm_block_intercomm);
		}
		MPI_Barrier(jm_block_comm);
		if(jm_block_rank == 0) {
			jm_log(1, "Distributing node names from worker");
		}
		// distribute to rest of jm_master in block
		MPI_Bcast(work_locbuf, locbufsize, MPI_CHAR, 0, jm_block_comm);
		if(jm_block_rank == 0) {
			jm_log("logfile '%s'\n", logfile);
		}
		MPI_Barrier(jm_block_comm);
		get_spawn_pids();
	}

	//
	// Print out the ranks and pids for this job
	//
	void print_pids_old() {
		if(jm_block_rank == 0) {
			char buf[32];
			std::string pidbuf = "    ";
			for(int i = 0; i < proccnt; i++) {
				if(i) {
					if(i%8 == 0) pidbuf += ",\n    ";
					else pidbuf += ", ";
				}
				pidbuf += jm_block_procname(pids[i].brank);
				pidbuf += ":";
				sprintf(buf, "%lu", (unsigned long)pids[i].pid);
				pidbuf += buf;
			}
			jm_log("Spawned pids for %s(%d):\n", jobname, jobid);
			printf("%s\n", pidbuf.c_str());
		}
	}
	std::string pid_range_str(std::string &proc, int startpid, int endpid) const {
		char buf[32];
		std::string r = proc;
		r += '[';
		sprintf(buf, "%d", startpid);
		r += buf;
		if(startpid != endpid) {
			sprintf(buf, "%d", endpid);
			r += '-';
			r += buf;
		}
		r += ']';
		return r;
	}
	void print_pids() {
		if(jm_block_rank == 0) {
			std::string pidbuf = "";
			std::string lastpname = "";
			int startpid = -1, endpid=-1;
			int numents = 0;
			for(int i = 0; i < proccnt; i++) {
				std::string pname = jm_block_procname(pids[i].brank);
				int pid = pids[i].pid;
				if(pname != lastpname || pid != endpid+1) {
					if(startpid >= 0) {
						if(numents++) pidbuf += ',';
						pidbuf += pid_range_str(lastpname, startpid, endpid);
					}
					// reset
					startpid = endpid = pid;
					lastpname = pname;
				} else {
					// lastpname must match
					endpid = pid; // extend range
				}
			}
			// flush last entry
			if(startpid >= 0) {
				if(numents++) pidbuf += ',';
				pidbuf += pid_range_str(lastpname, startpid, endpid);
			}
			jm_log("Spawned pids for %s(%d):\n", jobname, jobid);
			printf("   %s\n", pidbuf.c_str());
		}
	}

	// kill processes on all ranks in block that belong to job
	// The first time we will use SIGTERM which is catchable and
	// gives the task a chance to clean up before exit.   If it
	// doesn't die, then we upgrade to SIGKILL.
	void kill_job() {
		for(int i = 0; i < proccnt; i++) {
			if(pids[i].brank != jm_block_rank) continue;
			int pid = pids[i].pid;
			if(pid > 0) {
				// sig is SIGTERM at first, then we upgrade to SIGKILL
				// Diff is SIGTERM can be caught and ignored.  Should give proc a chance to exit nicely.
				// If proc ignores SIGTERM, then SIGKILL can't be ignored.
				if(kill(pid, pids[i].sig) < 0) {
					// ESRCH says that we couldn't find the process
					if(errno != ESRCH) {
						jm_log("Some issue with kill, errno=%d\n", errno);
					}
					if(pids[i].sig == SIGTERM) {
						pids[i].sig = SIGKILL;
					} else {
						// we delivered SIGKILL
						pids[i].pid = -1; // best we can do is assume that it is gone
					}
				}
			}
		}
	}

	// After a job is launched we give it a certain amount of time
	// to created <logfilestem>.started file.   Otherwise it is assumed
	// to have hung.   Example:  home dir can't be opened, or GPU fails
	// to init.
	void check_timeout() {
		struct stat st;
		int startinfo[2]; // startexists, fail
		//
		// if we are still looking for a start
		// start_timeout starts with some number like 60 (seconds)
		if(start_timeout != -1) {  // all ranks in lockstep in this if
			if(jm_block_rank == 0) {
				// only block 0 does the stat and time check.
				// can't risk ranks being out of sync
				if(stat(startfile, &st) == 0) { // if start file exists
					startinfo[0] = true;
					startinfo[1] = false;
					::unlink(startfile); // don't leave around
				} else { // still no start file.
					startinfo[0] = false;
					startinfo[1] = time(nullptr) > start_time + start_timeout;
				}
			}
			MPI_Bcast(&startinfo, 2, MPI_INT, 0, jm_block_comm);
			if(startinfo[0]) {
				start_timeout = -1; // start file written, all ranks record the fact that timeout was met.
				// we won't check again
			} else if(startinfo[1]) { // timeout failure
				if(jm_block_rank == 0) {
					printf("Master B%dR%d: job %s(%d) failed timeout.  Killing ...\n", jm_block_id, jm_block_rank, jobname, jobid);
				}
				kill_job();
				sleep(1); // time for kill to have effect?
			}
		}
	}

	// test a job to see if it is still active
	// Active requires two things
	//    The *.logtmp file still exists (not finished)
	//    all pids still exist.              (not crashed)
	// Return -1  :  Still running
	// Return 0   :  Finished with success
	// Return >0  :  return code
	int is_active() {
		int pcnt, bpcnt;
		int pid;
		int i;
		int rc;
		struct stat st;
		int logexists = 1;

		if(jm_block_rank == 0) {
			jm_log(2, "check job %s(%d) for activity\n", jobname, jobid);
		}
		if(jm_block_rank == 0) {
			logexists = stat(logfile, &st) == 0;
		}
		// let all jm_master ranks in block know
		MPI_Bcast(&logexists, 1, MPI_INT, 0, jm_block_comm);
		if(!logexists) 
			return 0; // must have been successful
		//
		// timeout is about the appearance of the <stemp>.start
		check_timeout();

		pcnt = bpcnt = 0;
		for(i = 0; i < proccnt; i++) {
			if(pids[i].brank != jm_block_rank) continue;
			pid = pids[i].pid;
			if(pid > 0) {
				rc = kill(pid, 0); // note sig==0, just checking if pid is active
				if(rc == 0) {
					// process exists
					bpcnt++;
				} else {
					// note:  this only happens on the rank that owns the pid
					pids[i].pid = -1; // process gone - no need to check again
				}
			}
		}
		MPI_Allreduce(&bpcnt, &pcnt, 1, MPI_INT, MPI_SUM, jm_block_comm);
		if(jm_block_rank == 0)
			jm_log(2, "job %s(%d) has %d of %d processes active\n", jobname, jobid, pcnt, proccnt);
		if(pcnt == proccnt)
			return -1;  // all ranks still running
		if(pcnt == 0)
			return 1;   // all ranks exited, but log file not renamed, error exit
		// In this case we have some of the ranks exited, but not all
		// We need to make sure they all go
		if(pcnt < lastproccnt) {
			lastproccnt = pcnt;
		} else {
			lastproccnt = pcnt;
			// no progress with exit - help out.
			// Each jm_master rank in block tries to kill job pids on its node
			// This is executed on all ranks.
			kill_job();
		}
		// allow to continue and check again
		return -1;
	}
};


// get name of processor parent block rank id is running on
char *jm_block_procname(int id) {
	if(id < 0 || id >= jm_block_size) err("bpid %d out of range", id);
	return jm_block_locbuf + id * JM_LOC_ENT_SIZE;
}

//
// Construct host tab for current block, removing duplicate host entries
// Called in Rank 0 of block
//
char **jm_make_host_tab() {
	int i, j, k;
	char *cp;
	SETLINE();
	char **tab = new char*[jm_block_size+1];

	for(i = 0; i <= jm_block_size; i++) tab[i] = nullptr;
	j = 0;
	for(i = 0; i < jm_block_size; i++) {
		cp = &jm_block_locbuf[i * JM_LOC_ENT_SIZE];
		for(k = j; --k >= 0;) {
			if(streq(cp, tab[k])) break;
		}
		if(k < 0) { // didn't find
			tab[j++] = jm_mstr(cp);
		}
	}
	if(j != jm_block_size) {
		jm_log("Wrong number (%d) of unique nodes for block of size %d\n", j, jm_block_size);
		for(i = 0; i < jm_block_size; i++) {
			cp = &jm_block_locbuf[i * JM_LOC_ENT_SIZE];
			printf("%d: %s\n", i, cp);
		}
		jm_log("Incorrect mpirun args?   Try --map-by node for initial launch\n");
		sleep(5);
		MPI_Abort(MPI_COMM_WORLD, 27);
	}
	return tab;
}

//
// free up host table after printing
//
void jm_free_host_tab(char **tab) {
	for(int i = 0; tab[i]; i++) {
		delete[] tab[i];
	}
	delete [] tab;
}

// We want to split the available nodes into blocks
void jm_split_blocks() {
	int i, hostlen;
	char *cp;

	if(jm_world_size % jm_block_size != 0) err("Expecting initial world size %d to be multiple of %d", jm_world_size, jm_block_size);
	jm_block_id = jm_world_rank / jm_block_size;
	jm_block_rank = jm_world_rank % jm_block_size;
	MPI_Comm_split(MPI_COMM_WORLD, jm_block_id, jm_block_rank, &jm_block_comm);
	int brank;
	MPI_Comm_rank(jm_block_comm, &brank);
	if(brank != jm_block_rank) err("Error on block rank");
	// now get table of processor names for the block
	int len;
	MPI_Get_processor_name(jm_this_procname, &len);
	jm_log("Processor '%s'\n", jm_this_procname);
	int locbufsize = jm_block_size * JM_LOC_ENT_SIZE;
	if(jm_block_rank == 0) {
		jm_log("jm_block_size=%d, locbufsize=%d\n", jm_block_size, locbufsize);
	}
	SETLINE();
	jm_block_locbuf = new char [locbufsize];
	MPI_Gather(jm_this_procname, JM_LOC_ENT_SIZE, MPI_CHAR, jm_block_locbuf, JM_LOC_ENT_SIZE, MPI_CHAR, 0, jm_block_comm);
	if(jm_block_rank == 0) {
		jm_log(2, "Gathered all hostnames from jm_master in block\n");
	}
	// make sure everyone has the table
	MPI_Bcast(jm_block_locbuf, locbufsize, MPI_CHAR, 0, jm_block_comm);
	if(jm_block_rank == 0) {
		jm_log("Broadcasted hostnames to block\n");
	}
	hostlen = 0;
	for(i = 0; i < jm_block_size; i++) {
		cp = &jm_block_locbuf[i * JM_LOC_ENT_SIZE];
		hostlen += jm_node_proccnt * (strlen(cp) + 1); // +1 for comma or final NUL
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

//
// jm_sched will be sending machine parameters to block 0 only.
// They are then broadcast within the Lump of jm_master processes
// 
void jm_readmachineparms() {
	int jm_slotenvbufsize;
	int mparms[JM_MACH_PARMS_SIZE];
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank == 0) {
		// only rank 0 gets machine parms from jm_sched
		jm_log(2, "Master B0R0: about to receive %d integers(parms of machine info) from jm_sched\n", JM_MACH_PARMS_SIZE); 
		MPI_Recv(mparms, JM_MACH_PARMS_SIZE, MPI_INT, 
			MPI_ANY_SOURCE, MPI_ANY_TAG, jm_sched_intercomm, MPI_STATUS_IGNORE);
		jm_log(2, "Master B0R0: received data\n");
		jm_block_size = mparms[1];    // number of nodes in block
		jm_node_maxwidth = mparms[2]; // threads on a node:  number of slots
		jm_slotenvbufsize = mparms[3]; // includes nul at end
		SETLINE();
		jm_slotenvbuf = new char[jm_slotenvbufsize];
		jm_log(2, "Master B0R0: about to receive %d chars for slotenvbuf\n", jm_slotenvbufsize);
		MPI_Recv(jm_slotenvbuf, jm_slotenvbufsize, MPI_CHAR, 
			MPI_ANY_SOURCE, MPI_ANY_TAG, jm_sched_intercomm, MPI_STATUS_IGNORE);
		jm_log("Received machine parameters from scheduler\n");
	}
	// broadcast to all jm_masters in this lump
	MPI_Bcast(mparms, JM_MACH_PARMS_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
	if(rank != 0) {
		// other blocks have to get copy too
		jm_block_size = mparms[1];    // number of nodes in block
		jm_node_maxwidth = mparms[2]; // threads on a node:  number of slots
		jm_slotenvbufsize = mparms[3];
		SETLINE();
		jm_slotenvbuf = new char[jm_slotenvbufsize];
	}
	// give everyone the slot based environment vars
	MPI_Bcast(jm_slotenvbuf, jm_slotenvbufsize, MPI_CHAR, 0, MPI_COMM_WORLD);

	if(rank == 0) {
		jm_log(2, "block size = %d, maxthread = %d\n", jm_block_size,jm_node_maxwidth);
		jm_log(2, "slotenvbuf=%s\n", jm_slotenvbuf);
	}
	jm_unpackslotenv(); // convert to slot based table.
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0) {
		jm_log("Done with machine parameters\n");
	}
}

//
// create a scheduler process somewhere
// that we can communicate with.
//
void jm_launch_scheduler(int argc, char *argv[]) {
	jm_sched_intercomm = 0;
	if(jm_world_rank != 0) return;
	int errcodes[1];
	int i;
	int rc;

	if(jm_world_rank == 0)
		jm_log("Launching scheduler\n");

	char *usedpm = getenv("MV2_SUPPORT_DPM");
	printf("jm_sched_launch: MV2_SUPPORT_DPM=%s\n", usedpm ? usedpm : "<NOTSET>");
	printf("jm_sched_launch: jm_sched_path=%s\n", jm_sched_path);

	int hlen;
	static char hostname[1024];

	MPI_Get_processor_name(hostname, &hlen);
	hostname[hlen] = 0;
	printf("jm_sched_launch: launch on node %s\n", hostname);

	//
	// sometimes PYTHONPATH doesn't get passed - force it.
	// Add JM_LAT_STARTUP_YAML as well
	//
	const char *env_pythonpath = "PYTHONPATH";
	const char *env_pyubase = "PYTHONUSERBASE";
	char *pythonpath = getenv(env_pythonpath);
	char *pyubase = getenv(env_pyubase);
	printf("jm_sched_launch: %s=%s\n", env_pythonpath, pythonpath ? pythonpath : "<NOTSET>");

	const char *env_jm_lat_startup_yaml = "JM_LAT_STARTUP_YAML";
	char *jm_lat_startup_yaml = getenv(env_jm_lat_startup_yaml);
	printf("jm_sched_launch: %s=%s\n", env_jm_lat_startup_yaml, jm_lat_startup_yaml ? jm_lat_startup_yaml : "<NOTSET>");
	if(!jm_lat_startup_yaml) {
		printf("jm_sched_launch: Missing specification of %s\n", env_jm_lat_startup_yaml);
		MPI_Abort(MPI_COMM_WORLD, 19);
	}

	{
		int proccnt = 1;
		MPI_Info infoa[3];
		char *cmds[1];
		char **argsa[1];
		int maxproc[1];
		vector<const char *> eargs; // first collect the args

		cmds[0] = jm_sched_path; // will process -env var value

		// specify host to launch jm_sched
		MPI_Info_create(&infoa[0]); // can use to pass env
		MPI_Info_set(infoa[0], "host", hostname);

		{
			// I think we have to pass PYTHONPATH this way
			// because python is managing to capture it before
			// jm_sched processes args
			char *envbuf = new char[strlen(env_pythonpath) + strlen(pythonpath) + 8];
			sprintf(envbuf, "%s=%s\n", env_pythonpath, pythonpath);
			MPI_Info_set(infoa[0], JM_MPI_SPAWN_ENV_VAL, envbuf);
			delete[] envbuf;
			if(pyubase) {
				// if set, also pass PYTHONUSERBASE
				envbuf = new char[strlen(env_pyubase) + strlen(pyubase) + 8];
				sprintf(envbuf, "%s=%s\n", env_pyubase, pyubase);
				// cray-mpich complained about '=' with "env" and said to use "env-val"
				// openmpi in 5.0.7 matches cray-mpich
				// openmpi in 4.1.8 uses "env"
				MPI_Info_set(infoa[0], JM_MPI_SPAWN_ENV_VAL, envbuf);
				delete[] envbuf;
			}
		}

		// Other environment can be passed on the command line.
		eargs.push_back("-env");
		eargs.push_back(env_jm_lat_startup_yaml);
		eargs.push_back(jm_lat_startup_yaml);

		eargs.push_back("-mpi"); // so jm_sched knows it's not a test run to load jobs

		printf("Adding passed args to eargs\n");
		for(i = 1; i < argc; i++) 
			eargs.push_back(argv[i]); // TODO:  should we quote?

		eargs.push_back(nullptr);

		for(unsigned i = 0; i < eargs.size(); i++) {
			printf("eargs[%d] = %s\n", i, eargs[i]);
		}

		maxproc[0] = proccnt;
		argsa[0] = (char **)&eargs[0]; // drop const
		rc = MPI_Comm_spawn_multiple(proccnt, cmds, argsa, maxproc, infoa, 0, 
			MPI_COMM_SELF, &jm_sched_intercomm, errcodes);
	}
	printf("jm_sched_launch: Returned from MPI_Comm_spawn rc=%d\n", rc);
	// MPI_Info_free(&info);
	if(errcodes[0] != MPI_SUCCESS) {
		err("Failed to launch scheduler");
	}

	if(jm_sched_intercomm == MPI_COMM_NULL) {
		err("Master R%d: jm_sched_intercomm is null!", jm_world_rank);
	} else {
		printf("Master R%d: jm_sched_intercomm is not null\n", jm_world_rank);
	}
	jm_log("Scheduler launched, args:\n");
	for(int i = 0; argv[i]; i++) {
		printf("   %d: %s\n", i, argv[i]);
	}
	jm_log(2, "About to Read back machine parameters\n");
	// now that spawn has happened, we have to fill in jm_sched_intercomm on
	// each jm_block_rank==0 process
}

//
//! \brief Connect add on block via block rank 0 intercomms to jm_sched.
// schedcomm is set in the first launch of jm_master because it spawns jm_sched
// Otherwise we pass MPI_COMM_NULL
//
static void jm_connect_scheduler(const char *hostname, bool done, MPI_Comm schedcomm) {
	const char *collectname = "mpijm";
	char link_port[MPI_MAX_PORT_NAME];
	MPI_Comm tmpcomm;
	int lumpsize;
	int rank;
	int rc;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// figure out lumpname
	char lumpname[LUMPNAMESIZE];
	strcpy(lumpname, hostname);

	if(rank == 0 && schedcomm == MPI_COMM_NULL) {
		// only need this in rank 0.
		MPI_Info lookinfo;
		MPI_Info_create(&lookinfo);
		MPI_Info_set(lookinfo, "ompi_lookup_order", "global"); // between mpiruns
		link_port[0] = 0;
		rc = MPI_Lookup_name(collectname, lookinfo, link_port);
		if(rc != MPI_SUCCESS) {
			err("Master: Failed to look up port with name %s for initial lump connection, rc=%d", collectname, (int)rc);
		}
		MPI_Info_free(&lookinfo);
		if(!link_port[0]) {
			err("Master: Failed to look up port with name %s for initial lump connection", collectname);
		}
		jm_log("Got link_port=%s to connect to\n", link_port);

		jm_log("Trying connection to %s\n", link_port);
		rc = MPI_Comm_connect(link_port, MPI_INFO_NULL, 0, MPI_COMM_SELF, &tmpcomm);
		if(rc != MPI_SUCCESS) {
			err("Master: lump connection to %s failed", link_port);
		}
		jm_log("Made connection, sending lumpname to scheduler\n");
		MPI_Send(lumpname, LUMPNAMESIZE, MPI_CHAR, 0, done? 0 : 1, tmpcomm);
		if(done) {
			jm_log("Sent message to end collection of lumps of nodes\n");
			MPI_Comm_disconnect(&tmpcomm);
			MPI_Finalize();
			exit(0);
		}
	} else {
		// this lump is the one that started jm_sched
		tmpcomm = schedcomm;
	}
	if(rank == 0) {
		lumpsize = jm_world_size;
		jm_log("lumpsize to scheduler\n");
		MPI_Send(&lumpsize, 1, MPI_INT, 0, 1, tmpcomm);
		jm_log("Reading nodesperblock from scheduler\n");
		MPI_Recv(&jm_block_size, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, tmpcomm, MPI_STATUS_IGNORE);
	}
	// Have to get one bit of machine info early so we can figure out blocks.
	MPI_Bcast(&jm_block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	jm_block_id = jm_world_rank / jm_block_size;
	jm_block_rank = jm_world_rank % jm_block_size;
	jm_block_count = jm_world_size / jm_block_size;

	if(rank == 0) {
		jm_log("Connecting blocks to scheduler\n");
	}
	// Now make private connections to first node of each block
	for(int bid = 0; bid < jm_block_count; bid++) {
		// get port info for the next block.
		if(rank == 0) {
			jm_log("Connecting block %d to scheduler\n", bid);
			MPI_Recv(link_port, MPI_MAX_PORT_NAME, MPI_CHAR,
				MPI_ANY_SOURCE, MPI_ANY_TAG, tmpcomm, MPI_STATUS_IGNORE);
			jm_log("Received private port info %s\n", link_port);
		}
		// tell all nodes in lump. Could just send to bid*jm_block_size
		MPI_Bcast(link_port, MPI_MAX_PORT_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
		if(rank == bid * jm_block_size) {
			// this is the jm_master to take the message
			rc = MPI_Comm_connect(link_port, MPI_INFO_NULL, 0, MPI_COMM_SELF, &jm_sched_intercomm);
			if(rc != MPI_SUCCESS) {
				err("Master: failed to connect to jm_sched on %s\n", link_port);
			}
			jm_log("Private connection established to block %d\n", bid);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if(rank == 0) {
		jm_log("Done connecting blocks to scheduler\n");
	}
}

//! \brief check to see if we are connecting to an existing scheduler
static bool jm_do_connect(int argc, char **argv) {
	for(int i = 1; i < argc; i++) {
		if(!strcmp(argv[i], "-connect"))
			return true;
	}
	return false;
}

//! \brief Check for flag that says to end collection of lumps.
static bool jm_end_connect(int argc, char **argv) {
	for(int i = 1; i < argc; i++) {
		if(!strcmp(argv[i], "-end"))
			return true;
	}
	return false;
}

static void malformedslotenv() {
	err("malformed slotenvbuf: %s\n", jm_slotenvbuf);
}

//
// jm_sched sent a set of slot level environment variables to pass
//
void jm_unpackslotenv() {
	char *cp, *tp;
	int ch;
	int slot;

	SETLINE();
	jm_slotenvlist = new jm_slotenv*[jm_node_maxwidth];
	for(int i = 0; i < jm_node_maxwidth; i++)
		jm_slotenvlist[i] = nullptr;

	// now parse the buffer into the slotlist table;
	char *name, *value;
	cp = jm_slotenvbuf;
	ch = *cp;
	while(ch) {
		if(*cp++ != '@') malformedslotenv();
		slot = atoi(cp);
		cp = strchr(cp, ':');
		if(!cp) malformedslotenv();
		cp++;
		tp = strchr(cp, '=');
		if(!tp) malformedslotenv();
		ch = *tp;
		*tp = 0;
		name = jm_mstr(cp);
		*tp = ch;
		cp = tp + 1;
		tp = cp;
		while(*tp && *tp != '@') tp++;
		ch = *tp;
		*tp = 0;
		value = jm_mstr(cp);
		*tp = ch;
		cp = tp;
		SETLINE();
		auto *sep = new jm_slotenv;
		sep->slot = slot;
		sep->name = name;
		sep->value = value;
		if(sep->slot < 0 || sep->slot >= jm_node_maxwidth)
			malformedslotenv();
		// save the environment variable in the table indexed by slot.
		jm_log(2, "saving slot=%d env %s=%s\n", sep->slot, sep->name, sep->value);
		sep->next = jm_slotenvlist[sep->slot];
		jm_slotenvlist[sep->slot] = sep;
	}

	// figure out how much padding we need to add slotenv entries on to the end of
	// the global environment vars in the spawn.
	jm_slotenv_xtra = 1;
	for(slot = 0; slot < jm_node_maxwidth; slot++) {
		int xsize = 1;
		for(jm_slotenv *sep = jm_slotenvlist[slot]; sep; sep=sep->next) {
			xsize += strlen(sep->name) + strlen(sep->value) + 4; // 4 is overkill, 2 should be enough
		}
		// find biggest extra amount
		if(xsize > jm_slotenv_xtra)
			jm_slotenv_xtra = xsize;
	}
}

//
// need to pack more info in jobbuf.
// program to launch
// arguments
//   Need to add these
// the number of processes to launch
// how to bind them to hardware resources
// environment variables
// TODO:  add start_timeout to jobbuf
// 
void jm_runjob(int *cmd, char *jobbuf) {
	char *pgm, *dir, *jobname, *logfile;
	vector<char *> args, env;
	jm_spjob *spjob;
	char *cp;
	char *envstr, *envtail;
	char *rankdata;
	int nthreadsperrank = 1;

	// Ready to launch
	// FIXME:  this proccnt is completely wrong
	// FIXME jm_block_proccnt = jm_block_size * jm_node_proccnt; // FIXME

	// extract job data for spawn
	pgm = dir = logfile = jobname = envstr = rankdata = nullptr;
	cp = jobbuf;
	while(1) {
		if(!strncmp(cp, JM_BUF_JOBNAME, JM_BUF_JOBNAME_LEN)) {
			cp += JM_BUF_JOBNAME_LEN;
			delete[] jobname;
			jobname = jm_mstr(cp);
			if(jm_block_rank == 0)
				jm_log(2, "Jobname %s\n", cp);
		} else if(!strncmp(cp, JM_BUF_PGM, JM_BUF_PGM_LEN)) {
			cp += JM_BUF_PGM_LEN;
			delete[] pgm;
			pgm = jm_mstr(cp);
			if(jm_block_rank == 0)
				jm_log(2, "pgm %s\n", cp);
		} else if(!strncmp(cp, JM_BUF_DIR, JM_BUF_DIR_LEN)) {
			cp += JM_BUF_DIR_LEN;
			delete[] dir;
			dir = jm_mstr(cp);
			if(jm_block_rank == 0)
				jm_log(2, "dir %s\n", cp);
		} else if(!strncmp(cp, JM_BUF_STDOUT, JM_BUF_STDOUT_LEN)) {
			cp += JM_BUF_STDOUT_LEN;
			delete[] logfile;
			logfile = jm_mstr(cp);
			if(jm_block_rank == 0)
				jm_log(2, "logfile %s\n", cp);
		} else if(!strncmp(cp, JM_BUF_ARG, JM_BUF_ARG_LEN)) {
			cp += JM_BUF_ARG_LEN;
			args.push_back(jm_mstr(cp));
			//if(jm_block_rank == 0)
			//	jm_log(2, "Arg %s\n", cp);
		} else if(!strncmp(cp, JM_BUF_ENV, JM_BUF_ENV_LEN)) {
			cp += JM_BUF_ENV_LEN;
			env.push_back(jm_mstr(cp));
			//if(jm_block_rank == 0)
			//	jm_log(2, "Env %s\n", cp);
		} else if(!strncmp(cp, JM_BUF_RANK, JM_BUF_RANK_LEN)) {
			cp += JM_BUF_RANK_LEN;
			rankdata = jm_mstr(cp);
			jm_log(2, "Received rank data '%s'\n", rankdata);
		} else if(!strncmp(cp, JM_BUF_NTHREAD, JM_BUF_NTHREAD_LEN)) {
			cp += JM_BUF_NTHREAD_LEN;
			nthreadsperrank = atoi(cp);
		} else if(!strncmp(cp, JM_BUF_END, JM_BUF_END_LEN)) {
			if(jm_block_rank == 0)
				jm_log(2, "End\n");
			break;
		} else {
			err("Unknown job data entry %s\n", cp);
		}
		// skip past NUL
		cp += strlen(cp) + 1;
	}
	args.push_back((char *)nullptr);
	// if(jm_block_rank == 0)
	// 	jm_log("Done extracting job data\n");
	if(!pgm) err("Missing program name");
	if(!jobname) err("Missing job name");
	// compute size of environment string
	MEMVERIFY();
	int i, esize;

	esize = 1 + jm_slotenv_xtra; // for NUL at end
	for(i = 0; i < (int)env.size(); i++) {
		esize += strlen(env[i]) + 1; // +1 for '\n' after each one
	}
	SETLINE();
	envstr = new char[esize+1];
	cp = envstr;
	for(i = 0; i < (int)env.size(); i++) {
		strcpy(cp, env[i]);
		cp += strlen(cp);
		*cp++ = '\n';
	}
	*cp = 0; // might be empty
	envtail = cp; // this is where we add on the slot specific env vars

	int proccnt = 1; // commas between slot ids
	for(cp = rankdata; *cp; cp++) {
		if(*cp == ',') proccnt++;
	}
	// proccnt has the number of ranks to launch

	// allocate a job object to track them
	// cmd[2] holds jobid from scheduler
	SETLINE();
	spjob = new jm_spjob(cmd[2], proccnt, nthreadsperrank, dir);

	MEMVERIFY();

	// FIXME: Mapping info would be added to info.
	// FIXME:  Figure out how to assign specific hardware

	int rc;
	int *slots = nullptr;
	if(jm_block_rank == 0) {
		SETLINE();
		int *ecodes = new int[proccnt];
		int i;
		// Do the actual launch
		SETLINE();
		slots = new int[proccnt];
		SETLINE();
		char **cmds = new char* [proccnt];
		char **dirs = new char* [proccnt];
		char **hosts = new char* [proccnt];
		SETLINE();
		char ***argva = new char ** [proccnt];
		SETLINE();
		int *maxprocs = new int [proccnt];
		SETLINE();
		MPI_Info *infoa = new MPI_Info [proccnt];
		char **htab = jm_make_host_tab();
		char *rd = rankdata;
		MEMVERIFY();

		for(i = 0; i < proccnt; i++) {
			// jm_log("Setting proc%d info\n", i);
			// figure out host and slot for this proc
			int bslot = atoi(rd); // "block" slot
			int node = bslot / jm_node_maxwidth; // find node from block slot
			slots[i] = bslot % jm_node_maxwidth;
			while(*rd && *rd != ',') rd++;
			if(*rd == ',') rd++; // for next number

			// jm_log("Setting proc%d pgm name\n", i);
			// TODO: Fix memory leak over args of launched program
#ifdef SPAWNWRAP
			cmds[i] = jm_mstr(SPAWNWRAPPGM);
			// vector<char *> args, env;
			// jm_log("Inserting -env args\n");
			vector<char *> eargs;
			for(int ei = 0; ei < (int)env.size(); ei++) {
				char *es = env[ei];
				char *cp = strchr(es, '=');
				if(!cp) continue;
				*cp = 0;
				// TODO: Check varname
				// jm_log("Adding job env %s=%s\n", es, cp + 1);
				eargs.push_back(jm_mstr("-env"));
				eargs.push_back(jm_mstr(es));
				eargs.push_back(jm_qstr(cp+1));
				*cp = '=';
			}
			// jm_log("Inserting -env args for slots\n");
			// Now for slot env vars
			for(jm_slotenv *sep = jm_slotenvlist[slots[i]]; sep; sep=sep->next) {
				// jm_log("Adding slot env %s=%s\n", sep->name, sep->value);
				eargs.push_back(jm_mstr("-env"));
				eargs.push_back(jm_mstr(sep->name));
				eargs.push_back(jm_qstr(sep->value));
			}
			// jm_log("Inserting pgm in jm_spawnwrap args\n");
			// now program and args
			eargs.push_back(jm_mstr(pgm));
			// jm_log("Inserting pgm args in jm_spawnwrap args\n");
			for(int ai = 0; args[ai];  ai++) {
				eargs.push_back(jm_mstr(args[ai]));
			}
			// jm_log("Cloning arg list\n");
			argva[i] = new char *[eargs.size() + 1];
			int esize = (int)eargs.size();
			for(int ai = 0; ai < esize; ai++) 
				argva[i][ai] = eargs[ai];
			argva[i][esize] = nullptr;
#else
			cmds[i] = jm_mstr(pgm);
			argva[i] = args.data();
#endif
			maxprocs[i] = 1;
			MPI_Info_create(&infoa[i]);
#ifndef SPAWNWRAP
			// append slot level env to global env list.  Space has been reserved.
			// jm_log(1, "append slot level env to global env list\n");
			cp = envtail;
			for(jm_slotenv *sep = jm_slotenvlist[slots[i]]; sep; sep=sep->next) {
				// jm_log(1,"Appending %s=%s\n", sep->name, sep->value);
				sprintf(cp, "%s=%s\n", sep->name, sep->value);
				cp += strlen(cp);
			}
			if(*envstr) { // if there is env to send, add to info
				// jm_log(1, "MPI_Info_set env[%d]=%s\n", i, envstr);
				MPI_Info_set(infoa[i], JM_MPI_SPAWN_ENV_VAL, envstr);  // MPI_Info_set does a strdup of the value
			}
#endif
			*envtail = 0; // trim environment back to global stuff, cutting off the slot env vars.

			if(dir) {
				dirs[i] = jm_mstr(dir);
				// jm_log(1, "MPI_Info_set wdir[%d]=%s\n", i, dirs[i]);
				MPI_Info_set(infoa[i], "wdir", dirs[i]);  // MPI_Info_set does a strdup of the value
			}
			//MPI_Info_set(infoa[i], "hostfile", jm_bhostfile);
			char *hcp = htab[node];
			hosts[i] = jm_mstr(hcp);
			// Section 10.3.4 of MPI 3.1 spec says this key is a hostname
			MPI_Info_set(infoa[i], "host", hosts[i]);
			// jm_log(1, "MPI_Info_set host[%d]=%s\n", i, hosts[i]);
		}
		MEMVERIFY();
		jm_free_host_tab(htab);

		if(jm_block_rank == 0)
			jm_log("Spawn of %d ranks of %s\n", proccnt, cmds[0]);

		MEMVERIFY();
		rc = MPI_Comm_spawn_multiple(proccnt, cmds, argva, maxprocs, infoa, 
			0, MPI_COMM_SELF, &jm_block_intercomm, ecodes);

		for(i = 0; i < proccnt; i++) {
			delete[] cmds[i];  // each rank got it's own allocated copy
			delete[] dirs[i];
			delete[] hosts[i];
			MPI_Info_free(&infoa[i]);
		}
		delete[] argva; // don't free individual tables, replicated from args.data()
		delete[] cmds;
		delete[] dirs;
		delete[] hosts;
		delete[] maxprocs;
		delete[] infoa;
		delete[] ecodes;
		MEMVERIFY();
		if(rc != MPI_SUCCESS) {
			char estring[MPI_MAX_ERROR_STRING];
			int elen;
			MPI_Error_string(rc, estring, &elen);
			jm_log("Failed to spawn job %s(%d) due to error %s", spjob->jobname, spjob->jobid, estring);
			rc = 1;
		}
		if(jm_block_rank == 0)
		jm_log("Spawn of %s for job %s(%d) complete\n", pgm, spjob->jobname, spjob->jobid);
	}
	// all block ranks get to know the return code
	MPI_Bcast(&rc, 1, MPI_INT, 0, jm_block_comm);

	spjob->setjobname(jobname);
	if(logfile) spjob->setlogfile(logfile);
	delete[] envstr;
	delete[] pgm;    
	delete[] dir;
	delete[] jobname;
	envstr = nullptr;
	pgm = nullptr;
	dir = nullptr;
	jobname = nullptr;
	MEMVERIFY();

	if(rc != 0) {
		spjob->finished(rc);
		return;
	}
	// let jm_sched know that the job has started
	spjob->started();

	// collect pid info
	spjob->send_parent_info(slots);
	delete[] slots;
	slots = nullptr;
	spjob->print_pids();
	// immedate disconnect of intercomm
	if(jm_block_rank == 0) {
#ifdef JM_DISCONNECT_CHILD
		MPI_Comm_disconnect(&jm_block_intercomm);
#endif
		// From here we monitor the child via the pids
		// under each block member by occasional polling
		jm_log("Done launching job %s(%d)\n", spjob->jobname, spjob->jobid);
	}
	spjob->start_time = time(nullptr);
	MEMVERIFY();
}

void jm_get_welcome() {
	char buf[JM_WELCOME_SIZE];

	if(jm_block_rank == 0) {
		if(jm_sched_intercomm == MPI_COMM_NULL) {
			err("jm_get_welcom: jm_sched_intercomm is NULL!");
		}
		MPI_Recv(buf, JM_WELCOME_SIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, jm_sched_intercomm, MPI_STATUS_IGNORE);
		jm_log("Received welcome message '%s'\n", buf);
	}
}

// basic MPI async programming
// http://www.mathcs.emory.edu/~cheung/Courses/561/Syllabus/92-MPI/async.html
void jm_sched_listen() {
	MPI_Request cmdrequest, jobrequest;
	int cmdflag;
	MPI_Status cmdstatus, jobstatus;
	int cmd[4], mcmd[4];
	char *jobbuf;
	int argbuflen;
	int i;
	jm_spjob *spjobnxt, *spjob;

	jm_get_welcome();

	if(jm_block_rank == 0) {
		// non blocking receive
		jm_log("Listening to sched\n");
		MPI_Irecv(&mcmd, 4, MPI_INT, 0, JM_CMD_TAG, jm_sched_intercomm, &cmdrequest);
	}
	while(1) {
		if(jm_block_rank == 0) {
			jm_log(2, "At listen loop top\n");
			MPI_Test(&cmdrequest, &cmdflag, &cmdstatus);
			if(cmdflag) {
				jm_log(2, "Received command %d:%d:%d\n", mcmd[0], mcmd[1], mcmd[2]);
				for(i = 0; i < 4; i++) cmd[i] = mcmd[i];
			} else {
				cmd[0] = JM_CMD_NAP;
				for(i = 1; i < 4; i++) cmd[i] = 0;
			}
		}
		// let rest of block know what we are doing
		MPI_Bcast(&cmd, 4, MPI_INT, 0, jm_block_comm);
		MEMVERIFY();
		switch(cmd[0]) {
		case JM_CMD_NAP:
			sleep(2);
			if(jm_block_rank == 0)
				jm_log(2, "Napping\n");
			break;
		case JM_CMD_JOB:
			argbuflen = cmd[1];
			if(argbuflen <= 0) {
				jm_log("Error: argbuflen=%d\n", argbuflen);
				MPI_Finalize();
				exit(1);
			}
			// now get command line, cmd[1] gives msg length
			SETLINE();
			jobbuf = new char[argbuflen+1];
			if(jm_block_rank == 0) {
				jm_log(2, "Received job command, looking for arguments\n");
				MPI_Irecv(jobbuf, argbuflen, MPI_CHAR, 0, JM_ARG_TAG, jm_sched_intercomm, &jobrequest);
				MPI_Wait(&jobrequest, &jobstatus);
			}
			MPI_Bcast(jobbuf, argbuflen, MPI_CHAR, 0, jm_block_comm); 
			if(jm_block_rank == 0) {
				jm_log("Received args (%d bytes)\n", argbuflen);
				jm_log("");
				for(int i = 0; i < argbuflen; i++) {
					int ch = jobbuf[i];
					printf("%c", (ch == 0) ? '@' : ch);
				}
				printf("\n");
			}
			// jobbuf should have a set of null terminated strings with an extra null at the end.
			jm_runjob(cmd, jobbuf);
			// restart command request
			if(jm_block_rank == 0) {
				MPI_Irecv(&mcmd, 4, MPI_INT, 0, JM_CMD_TAG, jm_sched_intercomm, &cmdrequest);
			}
			continue;
		case JM_CMD_QUIT:
			jm_log("Trying to quit\n");
			MPI_Finalize();
			if(jm_block_rank == 0)
				jm_log("Finalize completed\n");
			MEMVERIFY();
			exit(0);
			break;
		default:
			jm_log("Unknown command %d\n", cmd[0]);
			exit(1);
		}
		// now test if any jobs have gone inactive
		spjobnxt = jm_spjoblist;
		while(spjobnxt) {
			spjob = spjobnxt;
			spjobnxt = spjob->next;
			if(jm_block_rank == 0) {
				jm_log(2,"Checking for job %s(%d) completion\n", spjob->jobname, spjob->jobid);
			}
			int rc = spjob->is_active();
			if(rc < 0) continue; // still running

			// FIXME - go through pid rc values and set completionstatus

			// there are outstanding status requests for each submitted job
			// we just send it back with the right tag
			spjob->finished(rc);  // 0: success, >0: some kind of error
			delete spjob; // will unlink completed job
		}
	}
}

//! \brief check for -v argument indicating that jm_master should jm_log messages.
void jm_check_verbose(int argc, char **argv) {
	for(int i = 1; i < argc; i++) {
		if(!strcmp(argv[i], "-v"))
			jm_verbose = 1;
		if(!strcmp(argv[i], "-vv"))
			jm_verbose = 2;
	}
}

void jm_check_usage(int argc, char **argv) {
	if(argc != 2) return;
	char *s = argv[1];
	if(s[0] == '-' && s[1] == '-') s++;
	if(!(!strcmp(s, "-help") || !strcmp(s, "-usage"))) 
		return;
	printf("jm_master - control/monitoring process placed 1 per node to spawn and report back to scheduler\n");
	printf("usage:  jm_master -py <module> <args...>\n");
	printf("   module is the name of a module to load and call schedInit.  <args...> are extra arguments\n");
	printf("   passed in sys.argc,sys.argv\n");
	printf("usage:  jm_master [-connect] [-end]\n");
	printf("  -connect indicates that this set of jm_master should connect to a running scheduler\n");
	printf("  -end     indicates that the scheduler should stop collecting lumps of nodes and start running\n");
	exit(0);
}

// Log the command line
static void log_cmd(int argc, char *argv[]) {
	unsigned len = 20;
	for(int i = 0; i < argc; i++) len += strlen(argv[i]) + 4;
	char *buf = new char[len];
	strcpy(buf, "cmd: ");
	char *cp = buf + strlen(buf);
	for(int i = 0; i < argc; i++) {
		if(i) {
			*cp++ = ',';
			*cp++ = ' ';
		}
		unsigned alen = strlen(argv[i]);
		strcpy(cp, argv[i]);
		cp += alen;
	}
	printf("%s\n", buf);
	// jm_log(buf);
	delete[] buf;
}

// #include <mcheck.h>

int main(int argc, char *argv[]
// On the Mac environ is not a global variable, but is passed as an arg to main
#ifdef __APPLE__
	, const char **environ
#endif
) { 
    jm_setlinebuf(stdout);
	printf("jm_master startup\n");
	PNewInit();
	jm_check_usage(argc, argv);
	jm_check_verbose(argc, argv);
	bool doconnect = jm_do_connect(argc, argv);
	bool endconnect = jm_end_connect(argc, argv);
	static char hostname[128];
	gethostname(hostname, 120);
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &jm_world_size); 
	MPI_Comm_rank(MPI_COMM_WORLD, &jm_world_rank);
	printf("WR%d: jm_master startup pid=%ld on %s\n", jm_world_rank, (long)getpid(), hostname);
    jm_log("WR%d: jm_world_size = %d\n", jm_world_rank, jm_world_size);
	log_cmd(argc, argv);


	jm_sched_path = new char[strlen(argv[0]) + 20];
	jm_spawnwrap_path = new char[strlen(argv[0]) + 20];
	strcpy(jm_sched_path, argv[0]);
	char *cp = strrchr(jm_sched_path, '/');
	if(cp) {
		sprintf(cp, "/jm_sched");
	} else {
		strcpy(jm_sched_path, "jm_sched"); // hope it is in the path
		// TODO: check that jm_sched is in the path
	}

	strcpy(jm_spawnwrap_path, argv[0]);
	cp = strrchr(jm_spawnwrap_path, '/');
	if(cp) {
		sprintf(cp, "/jm_spawnwrap");
	} else {
		strcpy(jm_spawnwrap_path, "jm_spawnwrap"); // hope it is in the path
	}

	if(endconnect && !doconnect) {
		// send message to stop collecting
		jm_connect_scheduler(hostname, true, MPI_COMM_NULL);
	} else if(doconnect) {
		// new form where we connect to an existing scheduler
		jm_connect_scheduler(hostname, false, MPI_COMM_NULL);
		if(endconnect)
			jm_connect_scheduler(hostname, true, MPI_COMM_NULL);
	} else {
		jm_log("original form jm_launch_scheduler\n");
		// original form where we launch a scheduler
		jm_launch_scheduler(argc, argv);
		// Use jm_sched_intercomm as the initial link
		// instead of using the name server
		jm_connect_scheduler(hostname, true, jm_sched_intercomm);
	}
	MEMVERIFY();

	jm_readmachineparms();
	// scheduler invokes python code that figures out
	// machine parameters.  split blocks after scheduler launch.
	MEMVERIFY();
	jm_split_blocks();

	if(jm_block_rank == 0)
		jm_log("About to start listening to scheduler\n");
	MEMVERIFY();
	jm_sched_listen();

	// Don't reach here
	return 0; 
}


// Notes:

// MPI_COMM_SELF is a sub communicator where every process is in it's own
// group.
