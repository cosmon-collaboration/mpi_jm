#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"

// extern missing from stdio.h on summit
int	setlinebuf(FILE *stream);

int main(int argc, char **argv) {
	int rc = 0;
	MPI_Comm intercomm;
	char msg[] = "workers unite!";

	setlinebuf(stdout);
	MPI_Init(&argc, &argv);
	printf("In worker, doing work\n");
	printf("worker args:\n");
	for(int i = 1; i < argc; i++) {
		printf("worker argv[%d] = %s\n", i, argv[i]);
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
	}
	MPI_Finalize();
	exit(rc);
}
