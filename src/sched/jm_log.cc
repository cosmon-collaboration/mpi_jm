#include "jm_sched.h"

int jm_verbose = -1;

static char jm_logbuf[2048];
//
// Fatal Error reporting
//
void jm_err(const char *msg, ...) {
	char *buf = jm_logbuf;
	va_list args;
	va_start(args, msg);
	vsprintf(buf, msg, args);
	va_end(args);
	printf("Sched: %s", buf);
	MPI_Abort(MPI_COMM_WORLD, 17);
}

void jm_log(const char *fmt, ...) {
	if(1 > jm_verbose) return;
	char *buf = jm_logbuf;
	va_list args;
	va_start(args, fmt);
	vsprintf(buf, fmt, args);
	va_end(args);
	printf("Sched: %s", buf);
}

void jm_log(int lvl, const char *fmt, ...) {
	if(lvl > jm_verbose) return;
	char *buf = jm_logbuf;
	va_list args;
	va_start(args, fmt);
	vsprintf(buf, fmt, args);
	va_end(args);
	printf("Sched %s",  buf);
}
