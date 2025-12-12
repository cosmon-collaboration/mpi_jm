//
// Perform spawn and try to survive abort of child after spawn
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "mpi.h"

// strdup missing somehow
char *strdup(const char *s) {
	char *buf = (char *)malloc(strlen(s) + 1);
	strcpy(buf, s);
	return buf;
}

void *zalloc(size_t sz) {
	void *buf = malloc(sz);
	memset(buf, 0, sz);
	return buf;
}

int main(int argc, char **argv) {
	int k;
	int rank;
	int rc;
	int errcodes[10];
	char *sargv[argc+4];
	MPI_Comm intercomm = MPI_COMM_NULL;
	char *workpath, *wrappath, *cp;
	int hlen;
	static char hostname[1024]; // look up max proc name

	setvbuf(stdout, NULL, _IOLBF, 0); // flush lines as they are written
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Get_processor_name(hostname, &hlen);
	hostname[hlen] = 0; // make sure

	printf("Master R%d@%s master pgm: %s", rank, hostname, argv[0]);
	for(k = 1; k < argc; k++) printf(" %s", argv[k]);
	printf("\n");

	sleep(2);
	int plen = strlen(argv[0]) + 20;
	workpath = zalloc(plen);
	wrappath = zalloc(plen);
	strcpy(workpath, argv[0]);
	strcpy(wrappath, argv[0]);
	cp = strrchr(workpath, '/');
	if(!cp) {
		MPI_Finalize();
		printf("run master as ./master, or full path\n");
		exit(1);
	}
	strcpy(cp, "/worker");
	cp = strrchr(wrappath, '/');
	strcpy(cp, "/jm_spawnwrap");
	printf("Master R%d: worker pgm: %s\n", rank, workpath);
	printf("Master R%d: Thinking about spawn\n", rank);
	printf("spawnwrap path: %s\n", wrappath);
	printf("worker path: %s\n", workpath);
	sargv[0] = zalloc(10);
	strcpy(sargv[0], "-mpi");
	for(k = 1; k < argc; k++) sargv[k] = argv[k];
	sargv[k] = NULL; // end spawn args with NULL

	char *usedpm = getenv("MV2_SUPPORT_DPM");
	printf("MV2_SUPPORT_DPM=%s\n", usedpm ? usedpm : "<NOTSET>");
	fflush(stdout);

#define USE_SPAWN_MULTIPLE
#ifdef USE_SPAWN_MULTIPLE
	{
		int proccnt = 1;
		MPI_Info *infoa =  (MPI_Info *)zalloc((proccnt+2) * sizeof(MPI_Info));
		char **cmds = (char **)zalloc(proccnt * sizeof(char *));
		char ***args = (char ***)zalloc(3 * sizeof(char **));
		char **env = (char **)zalloc(proccnt * sizeof(char *));
		char **hosts = (char **)zalloc(proccnt * sizeof(char *));
		int *maxproc = (int *)zalloc(proccnt * sizeof(int));
		for(int i = 0; i < proccnt; i++) {
			cmds[i] = strdup(wrappath);

			MPI_Info_create(&infoa[i]);
			// MPI_Info_set(infoa[i], "env", strdup("FOO='hello'"));
			MPI_Info_set(infoa[i], "PMIX_ENVAR", strdup("FOO='hello'"));
			MPI_Info_set(infoa[i], "wdir", ".");
			hosts[i] = strdup(hostname);
			MPI_Info_set(infoa[i], "host", hosts[i]);
			MPI_Info_set(infoa[i], "rankpool", "all_nodes");

			args[i] = (char **)zalloc(15 * sizeof(char **));
			args[i][0] = strdup("-env");
			args[i][1] = strdup("FOO");
			args[i][2] = strdup("bar");
			args[i][3] = strdup(workpath);
            for( k = 0; sargv[k]; k++ )
                args[i][k+4] = strdup(sargv[k]);
			args[i][k+4] = 0;
			maxproc[i] = 1;
		}
		rc = MPI_Comm_spawn_multiple(proccnt, cmds, args, maxproc, infoa, 0, MPI_COMM_SELF, &intercomm, errcodes);
		for(int i = 0; i < proccnt; i++) {
			free(args[i][0]);
			free(args[i][1]);
			free(args[i]);
			free(cmds[i]);
			free(env[i]);
			free(hosts[i]);
		}
		free(infoa);
		free(cmds);
		free(args);
		free(env);
		free(hosts);
	}
#else
	rc = MPI_Comm_spawn(workpath, sargv, 2, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes);
#endif
	printf("spawn rc=%d\n", rc);

	free(workpath);
	workpath = NULL;

	printf("master R%d: launched worker!\n", rank);

	int msgsize;
	char *buf;

	MPI_Bcast(&msgsize, 1, MPI_INT, 0, intercomm);
	printf("master R%d: received: msgsize=%d\n", rank, msgsize);
	buf = zalloc(msgsize + 1);
	MPI_Bcast(buf, msgsize, MPI_CHAR, 0, intercomm);
	printf("master R%d: received: %s\n", rank, buf);
	free(buf);
	MPI_Comm_disconnect(&intercomm);
	for(int i = 0; i < 10; i++) {
		sleep(1);
		printf("Master still here %d\n", i);
	}
	printf("Master calling Finalize - Disconnect must have worked\n");
	MPI_Finalize();
	exit(0);
}
