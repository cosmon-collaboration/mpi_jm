#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "mpi.h"

#define TRUE 1
#define FALSE 0
// extern missing from stdio.h on summit
// pointer for  testing segv
int *zp = 0;

int main(int argc, char **argv) {
	int rc = 0;
	MPI_Comm intercomm;
	char msg[] = "workers unite!";

	setvbuf(stdout, NULL, _IOLBF, 0); // flush lines as they are written
	MPI_Init(&argc, &argv);
	printf("In worker, doing work\n");
	printf("worker args:\n");
    int dosegv=FALSE;
	for(int i = 1; i < argc; i++) {
		printf("worker argv[%d] = %s\n", i, argv[i]);
        if( ! strcmp ( argv[i], "-segv"))
            dosegv = TRUE;
	}

	MPI_Comm_get_parent(&intercomm);
	if(intercomm == MPI_COMM_NULL) {
		printf("worker: No parent!\n");
		rc = 1;
	} else {
		int psize;
		int msgsize;

		printf("worker: got parent\n");
		MPI_Comm_remote_size(intercomm, &psize);
		printf("worker: master has %d ranks\n", psize);
		printf("worker: sending msg string\n");
		msgsize = strlen(msg) + 1;
		MPI_Bcast(&msgsize, 1, MPI_INT, MPI_ROOT, intercomm);
		MPI_Bcast(msg, msgsize, MPI_CHAR, MPI_ROOT, intercomm);

		// checking that env from spawn propagated.
		char *foo = getenv("FOO");
		if(!foo) foo = "<unset>";
		char *bar = getenv("BAR");
		if(!bar) bar = "<unset>";
		printf("env test\n");
		printf("   FOO=\"%s\"\n", foo);
		printf("   BAR=\"%s\"\n", bar);

		printf("worker: disconnecting from intercomm to prevent error propagation to master\n");
		MPI_Comm_disconnect(&intercomm);
		printf("worker: disconnected from intercomm\n");
		sleep(2);
        if( dosegv ){
            printf("worker: testing segv with disconnect\n");
            *zp = 1;
        } else {
            printf("worker: aborting to test disconnect\n");
            MPI_Abort(MPI_COMM_WORLD, 7);
        }
	}
	MPI_Finalize();
	exit(rc);
}
