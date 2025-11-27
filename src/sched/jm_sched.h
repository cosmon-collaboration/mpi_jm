#include <Python.h>
#include "structmember.h"
#include <unistd.h>
#include "mpi.h"
#include <cstdlib>
#include <cstdarg>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <ratio>
#include <chrono>
#include <vector>
using std::vector;
#include "jm.h"
#include "jm_int.h"
#include "jm_job.h"
#include "jm_res.h"
#include "jm_py.h"

extern int jm_verbose;

char *jm_mstr(const char *);
void jm_sched_abort();
void jm_err(const char *msg, ...);
void jm_log(const char *fmt, ...);
void jm_log(int lvl, const char *fmt, ...);
