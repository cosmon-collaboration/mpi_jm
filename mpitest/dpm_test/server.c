//
// The idea here is to launch one copy of this program.
// Then we do mpirun of the 
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <string.h>
#include "mpi.h"
#include "smcomm.h"

// extern missing from stdio.h on summit
int setlinebuf (FILE *stream);

static char servicename[128];
static char port_name[MPI_MAX_PORT_NAME];

void do_publish() {
	MPI_Info publishinfo;
	MPI_Info_create(&publishinfo);
	MPI_Info_set(publishinfo, "ompi_global_scope", "true");
	MPI_Publish_name(servicename, publishinfo, port_name);
	MPI_Info_free(&publishinfo);
	printf("Server: published name %s.%s\n", servicename, port_name);
}

void do_unpublish() {
	MPI_Info publishinfo;
	MPI_Info_create(&publishinfo);
	MPI_Info_set(publishinfo, "ompi_global_scope", "true");
	MPI_Unpublish_name(servicename, publishinfo, port_name);
	MPI_Info_free(&publishinfo);
}

//
// AcceptBlocks will run in a separate thread
//
static void AcceptBlocks(void) {
	MPI_Comm client;
	MPI_Status status;
	char blockname[MAXBLOCKNAME];
	int collecting = TRUE;

	while(collecting) {
		sleep(1);
		// need to treat as critical region.
		printf("trying accept on port_name=%s\n", port_name);
		MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &client); 
		printf("accepted connection\n");
		// do_publish();

		printf("receiving data\n");
		MPI_Recv(blockname, MAXBLOCKNAME, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, client, &status);
		printf("receive completed, tag=%d\n", (int)status.MPI_TAG);
		switch(status.MPI_TAG) {
		case 0:
			printf("Server: Received tag=0 from \'%s\', stop connecting\n", blockname);
			MPI_Barrier(client);
			MPI_Comm_disconnect(&client);
			printf("Disconnect from \'%s\' done\n", blockname);
			collecting = FALSE;
			break;
		case 1:
			printf("Server: received message from %s\n", blockname);
			// Must disconnect so we can accept a new connection
			MPI_Barrier(client);
			MPI_Comm_disconnect(&client);
			printf("Disconnect from \'%s\' done\n", blockname);
			break;
		default:
			printf("Unknown message tag=%d\n", (int)status.MPI_TAG);
			break;
		}
	}
	printf("Done collecting\n");
}

int main(int argc, char **argv) {
	int i, k;
	int rank, size;
	int errcodes[1];
	char *sargv[argc+4];
	MPI_Comm intercomm;
	char *workpath, *cp;

	setlinebuf(stdout);
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if(provided != MPI_THREAD_MULTIPLE && provided != MPI_THREAD_SERIALIZED)
		printf("Openmpi was not built to support at least MPI_THREAD_SERIALIZED, see MPI_Init_thread\n");
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(size != 1) {
		if(rank == 0)
			printf("Server: should have 1 rank");
	}
	printf("Server: pid=%d\n", (int) getpid());

	// this code needs to be in a thread so we don't block
	// linkport is used for blocks to report in on.  once they do we want to
	// switch to a private connection between the block and sched.

	// blocks first connect to the 
	// http://mpi-forum.org/docs/mpi-3.1/mpi31-report/node248.htm

// not ifdef
#if USE_MPI_OPEN_PORT
	// Open a port and publish the name to a file so clients can see it.
	unlink(PORTFILE); // get rid of old portfile as soon as possible.
    MPI_Open_port(MPI_INFO_NULL, port_name);
    printf("Server opened port: <%s>.\n", port_name);
	// write to a file.  The client will have to agree on file name.
	char *tmpname = (char *)malloc(strlen(PORTFILE) + 10);
	sprintf(tmpname, "%s_tmp", PORTFILE);
	// write to temp file and rename once complete.
	// client can wait for file to appear.
    FILE *fp = fopen(tmpname, "w");
	if(!fp) {
		printf("Can't open port file %s\n", tmpname);
		exit(1);
	}
	fprintf(fp, "%s", port_name);
	fclose(fp);
	rename(tmpname, PORTFILE);
#else
	// Create a port for another process to connect to
	strcpy(servicename, SERVICE_NAME);
	MPI_Info portinfo;
	MPI_Info_create(&portinfo);
	MPI_Open_port(portinfo, port_name);
	MPI_Info_free(&portinfo);
	printf("Server: servicename=%s, port_name=%s\n", servicename, port_name);
	do_publish();
#endif

	AcceptBlocks();

// not ifdef
#if USE_MPI_OPEN_PORT
	MPI_Close_port(port_name);
	unlink(PORTFILE);
#else
	do_unpublish();
#endif

	MPI_Finalize();
	exit(0);
}
