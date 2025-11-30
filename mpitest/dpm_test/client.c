//
// The idea here is to launch one copy of this program.
// Then we do mpirun of the 
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include "mpi.h"
#include "smcomm.h"

static char link_port[MPI_MAX_PORT_NAME];
// int setlinebuf(FILE *stream);
#define my_setlinebuf(stream) setvbuf(stream, NULL, _IOLBF, 0)

int main(int argc, char **argv) {
	int rank, size;
	MPI_Comm server;
	int tag;
	char blockname[MAXBLOCKNAME];
	char link_port[MPI_MAX_PORT_NAME];
	int provided;

	my_setlinebuf(stdout);

	sleep(2);

	sprintf(blockname, "blockname_%d", (int)getpid());

	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	printf("client B%d: MPI_Init done, pid=%d\n", rank, (int)getpid());


#if USE_MPI_OPEN_PORT
	int i;
	const int maxtries = 10;
	for(i = 0; i < maxtries; i++) {
		printf("%d: looking for port file %s\n", i, PORTFILE);
		FILE *fp = fopen(PORTFILE, "r");
		if(fp) {
			printf("Found port file %s\n", PORTFILE);
			// found it.
			int pos = 0;
			while(TRUE) {
				int ch = getc(fp);
				if(ch == EOF || ch == 0) break;
				if(pos >= sizeof(link_port)-1) {
					printf("FILE %s is too large for link_port\n", PORTFILE);
					exit(1);
				}
				link_port[pos++] = ch;
			}
			link_port[pos] = 0; // NUL terminate
			fclose(fp);
			break;
		}
		sleep(1); // wait and retry.
	}
	if(i >= maxtries) {
		printf("Didn't find port file %s.  Server failure??\n", PORTFILE);
		exit(1);
	}
#else
	MPI_Info lookinfo;
	MPI_Info_create(&lookinfo);
	MPI_Info_set(lookinfo, "ompi_lookup_order", "global");
	printf("Using lookup on %s\n", SERVICE_NAME);
	MPI_Lookup_name(SERVICE_NAME, lookinfo, link_port);
	MPI_Info_free(&lookinfo);
#endif
	printf("MAXBLOCKNAME=%d\n", MAXBLOCKNAME);
	printf("link_port = %s\n", link_port);
	printf("trying connection\n");
	
	MPI_Comm_connect(link_port ,MPI_INFO_NULL, 0, MPI_COMM_SELF, &server);

	printf("Connected!\n");
	fflush(stdout);

	tag = 1;
	int i = 1;
	while(i < argc) {
		if( !strcmp(argv[i], "-done")) {
			tag = 0;
			i++;
		} else if( (i + 1) < argc && !strcmp(argv[i], "-msg")) {
			strcpy(blockname, argv[i+1]);
			i+=2;
		} else {
			printf("Unknown argument '%s'\n", argv[i]);
			i++;
		}
	}
	printf("Sending %s, and tag=%d\n", blockname, tag);
	MPI_Send(blockname, MAXBLOCKNAME, MPI_CHAR, 0, tag, server );

	printf("Attempting disconnect from server\n");
	MPI_Barrier(server);
	MPI_Comm_disconnect(&server);
	printf("Disconnected from server\n");

	MPI_Finalize();
	exit(0);
}
