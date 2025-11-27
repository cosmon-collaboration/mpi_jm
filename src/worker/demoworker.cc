/* worker */ 
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <limits.h>
#include "mpi.h" 
#include "jm.h"
#include <execinfo.h>
#include <signal.h>

void handler(int sig) {
	void *array[20];
	size_t size;

	// get void*'s for all entries on the stack
	size = backtrace(array, 20);

	// print out all the frames to stderr
	fprintf(stderr, "Error: signal %d:\n", sig);
	backtrace_symbols_fd(array, size, STDERR_FILENO);
	exit(1);
}

void installhandlers() {
	signal(SIGSEGV, handler);
	signal(SIGBUS, handler);
}

int main(int argc, char *argv[]) 
{ 
	setlinebuf(stdout);
	printf("Entering worker\n");
	int rc = 0;
	int rank;
	char *procname = new char[MPI_MAX_PROCESSOR_NAME];
	char *dirname =  jm_getcwd();

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided); 
	//installhandlers();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int len;
	jm_parent_handshake(&argc, &argv);
	MPI_Get_processor_name(procname, &len);
	int cpuid = jm_getcpu();
	printf("Worker R%d: %s: After MPI_Init procname=%s, cpuid=%d, dir=%s\n",
		rank, jm_jobname, procname, cpuid, dirname);
	/* 
	 * Parallel code here.  
	 * The manager is represented as the process with rank 0 in (the remote 
	 * group of) MPI_COMM_PARENT.  If the workers need to communicate among 
	 * themselves, they can use MPI_COMM_WORLD. 
	 */ 
	printf("Worker R%d: '%s' doing important work\n", rank, jm_jobname);
	const char *fooenv = getenv("FOO");
	if(!fooenv) fooenv = "<missing>";
	const char *barenv = getenv("BAR");
	if(!barenv) barenv = "<missing>";
	printf("Work:%sR%d: FOO=%s\n", jm_jobname, rank, fooenv);
	printf("Work:%sR%d: BAR=%s\n", jm_jobname, rank, barenv);

	const char *emotion = getenv("EMOTION");
	if(!emotion) emotion = "<none>";
	printf("Work:%sR%d: EMOTION=%s\n", jm_jobname, rank, emotion);

	delete[] procname;
	delete[] dirname;

	printf("Work:%sR%d: About to Finalize\n", jm_jobname, rank);
	MPI_Finalize(); 
	printf("Work:%sR%d: Finalize Done\n", jm_jobname, rank);
	jm_finish(rc, "message");
	return 0; 
} 
