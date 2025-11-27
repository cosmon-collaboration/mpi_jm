/*
 * Scheduler.   A single copy of the process is run
 * somewhere in the available set of nodes.
 * A job file is read and then the scheduler
 * launches jobs to available blocks.  
 */
#include "jm_sched.h"
#include <algorithm>

// default is forever
uint64_t jm_alloc_time = (uint64_t)1 << 60;       // number of seconds requested in allocation
Lump *jm_parent_lump;
int jm_block_size_arg = -1;  // overrides default from machine file

// For testing we run sched without an MPI connection
// to see if job construction is correct.
bool jm_mpi_enable = false;
bool jm_mpi_init_done = false;
bool jm_collect_blocks = false;
const char *jm_loadmodname = nullptr;

// number of ranks in a block.   In normal use this is the
// number of nodes as we try to start with one process per
// node.

// keep track of what each block is up to.
// Indexed by block id.   The value is a 
// resource mask indicating the resources in use.
int jm_block_count;          // number of blocks
jm_block_use_t *jm_block_use = nullptr;   // length jm_block_count

static const char *jm_pyfile = nullptr;
wchar_t *jm_pyprogram = nullptr;
static int jm_numpyargs = 0;
static char **jm_pyargs = nullptr;

//! \brief Used when scheduler startup fails. i.e. job file not readable ...
void jm_sched_abort() {
	printf("Sched: Aborting\n");
	if(jm_mpi_init_done) {
		MPI_Abort(MPI_COMM_WORLD, 18);
		// should not return
	}
	exit(1);
}


//! \brief Constructor for block tracking class.  Mark block initially unused.
jm_block_use_t::jm_block_use_t() {
	int numres = jmres_block.numres();
	int bslotcnt = jmres_block.blockslots();
	int numnodes = jmres_block.getnumnodes();
	// during launch process lock is set.
#ifndef LUMPSPAWNLOCK
	lockval = 0; // set to jobid of current owner.
#endif
	rescnt = new int[numres];
	for(int i = 0; i < numres; i++) rescnt[i] = 0;
	slots = new int[bslotcnt]; 
	for(int i = 0; i < bslotcnt; i++) slots[i] = 0;
	nodememKb = new int[numnodes];
	for(int i = 0; i < numnodes; i++) nodememKb[i] = 0.0;
	lp = nullptr;
	lbid = -1;
}
// destructor
jm_block_use_t::~jm_block_use_t() {
	delete[] rescnt;
	delete[] slots;
}
// We would prefer to lock at the block level, but it looks like under mpiexec.hydra there can't be
// two spawns happening at the same time.
#ifdef LUMPSPAWNLOCK
void Lump::doublelock() {
	printf("Lump locked twice!\n");
	MPI_Abort(MPI_COMM_WORLD, 19);
}
void Lump::badunlock(int jobid) {
	printf("Lump locked by job %d, unlock attempted by job %d\n", lockval, jobid);
	MPI_Abort(MPI_COMM_WORLD, 19);
}
void jm_block_use_t::setLock(int jobid) { lp->setLock(jobid); }
bool jm_block_use_t::isLocked() { return lp->isLocked(); }
void jm_block_use_t::clrLock(int jobid) { lp->clrLock(jobid); }
#else
void jm_block_use_t::doublelock() {
	printf("Block locked twice!\n");
	MPI_Finalize();
	exit(19);
}
void jm_block_use_t::badunlock(int jobid) {
	printf("Block locked by %d, unlock attempted by %d\n", lockval, jobid);
	MPI_Finalize();
	exit(19);
}
void jm_block_use_t::setLock(int jobid) { if(lockval) doublelock(); lockval = jobid; }
bool jm_block_use_t::isLocked() { return lockval != 0; }
void jm_block_use_t::clrLock(int jobid) { if(lockval != jobid) badunlock(jobid); lockval = 0;}
#endif

int jm_block_use_t::blkUsedMemKb() {
	int mem = 0;
	int numnodes = jmres_block.getnumnodes();
	for(int i = 0; i < numnodes; i++) mem += nodememKb[i];
	return mem;
}

// Verify that all memory is marked unused.
// Called at end to make sure it all gets returned.
static void jm_verify_block_mem0() {
	bool err = false;
	for(int bi = 0; bi < jm_block_count; bi++) {
		int mem = jm_block_use[bi].blkUsedMemKb();
		if(mem > 0) {
			printf("block %d still has %dKb reserved\n", bi, mem);
			err = true;
		}
	}
	if(err) {
		printf("Block Memory Error!\n");
		MPI_Finalize();
		exit(19);
	}
}

//
// Call a python function with arguments and get a return value
//
static PyObject *callPythonFunc(const char *modname, const char *funcname, int argcnt, PyObject **args) {
	// convert from simple char * path to Python unicode object
	PyObject *pModule;
	// printf("Sched: Loading module '%s'\n" , modname);
	pModule = PyImport_ImportModule(modname);
	if(!pModule) {
		PyErr_Print();
		printf("Load of %s failed\n", modname);
		jm_sched_abort();
	}
	PyObject *pFunc = nullptr;
	const char *s = strchr(funcname, '.');
	if(s) {
		char *buf = new char[strlen(funcname) + 1];
		int len = s - funcname;
		for(int i = 0; i < len; i++) buf[i] = funcname[i];
		buf[len] = 0;
		PyObject *pModuleDict = PyModule_GetDict(pModule);
		PyObject *pClass = PyDict_GetItemString(pModuleDict, buf);
		if(!pClass) {
			printf("Can't find class %s\n", buf);
			jm_sched_abort();
		}
		pFunc = PyObject_GetAttrString(pClass, s+1);
		// look for our top level call to load jobs
		if(!pFunc) {
			printf("Can't find method %s.%s\n", modname, funcname);
			jm_sched_abort();
		}
	} else {
		pFunc = PyObject_GetAttrString(pModule, funcname);
		// look for our top level call to load jobs
		if(!pFunc) {
			printf("Can't find function %s.%s\n", modname, funcname);
			jm_sched_abort();
		}
	}
	if(!PyCallable_Check(pFunc)) {
		printf("%s.%s is not callable", modname, funcname);
		jm_sched_abort();
	}
	PyObject *pArgs = PyTuple_New(argcnt);
	for(int i = 0; i < argcnt; i++) {
		PyTuple_SetItem(pArgs, i, args[i]);
	}
	Py_INCREF(pArgs);
	//PyObject *pValue = PyLong_FromLong(1);
	//PyTuple_SetItem(pArgs, 0, pValue);
	jm_loadmodname = modname;
	PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
	jm_loadmodname = nullptr;
	Py_DECREF(pArgs);
	return pValue;
}

//
// Interface for startup calls
//
static void invokePythonScript(const char *modname, const char *funcname) {
	PyObject *pValue;
	const char *s = getenv("PYTHONPATH");
	if(!s) s = "<not set>";
	printf("Sched: PYTHONPATH=%s\n", s);
	s = getenv("PYTHONUSERBASE");
	if(!s) s = "<not set>";
	printf("Sched: PYTHONUSERBASE=%s\n", s);

	pValue = callPythonFunc(modname, funcname, 0, nullptr);
	if(pValue) {
		printf("Python call to %s.%s succeeded!\n", modname, funcname);
		Py_DECREF(pValue);
	} else {
		PyErr_Print();
		printf("Python call to %s.%s failed\n", modname, funcname);
		jm_sched_abort();
	}
}
// return a newly allocated copy of string s
char *jm_mstr(const char *s) {
	if(!s) {
		printf("Sched: jm_mstr of nil string");
		jm_sched_abort();
	}
	char *buf = new char[strlen(s) + 1];
	strcpy(buf,s);
	return buf;
}

//
// -blocksize and -alloctime are also passed to the startup script
// It is a hack, but dig them out first
//
void jm_arg_prescan(int argc, char *argv[]) {
	int lim = argc - 1;
	for(int i = 0; i < lim; i++) {
		if(streq(argv[i], "-blocksize")) {
			jm_block_size_arg = atoi(argv[i+1]);
			if(jm_block_size_arg <= 0) {
				printf("Sched: -blocksize must be positive integer\n");
				jm_sched_abort();
			}
		} else if(streq(argv[i], "-alloctime")) {
			int rc = sscanf(argv[i+1], "%" PRIu64, &jm_alloc_time);
			if(rc != 1) {
				printf("Sched: -alloctime has bad format\n");
				jm_sched_abort();
			}
			printf("Sched: alloctime = %" PRIu64 "\n", jm_alloc_time);
		}
	}
}

//
// Find job manager arguments that apply to the
// scheduler.
//
void jm_sched_parseargs(int argc, char *argv[]) {
	int pos;
	bool secondarg, thirdarg;
	const char *s, *s2;

	for(pos = 0; pos < argc; pos++){
		printf("Sched: arg[%d] = %s\n", pos, argv[pos]);
	}

	// prescan for -blocksize and -alloctime.  They are lso passed as part of -py <module> <extra...>
	jm_arg_prescan(argc, argv);

	pos = 1;
	while(pos < argc) {
		secondarg = (pos + 1) < argc;
		thirdarg = (pos + 2) < argc;
		s = argv[pos];
		if(s[0] != '-') {
			pos++;
			continue;
		}
		s2 = secondarg ? argv[pos+1] : "<empty>";
		if(thirdarg && streq(s, "-env")) {
			setenv(s2, argv[pos+2], 1/*overwrite*/);
			pos += 3;
		} else if(secondarg && streq(s, "-py")) { // consumes the rest of the args
			char *htp = jm_mstr(s2);
			// get last '.' in htp
			char *ltp = nullptr;
			for(char *tp = htp; *tp; tp++) {
				if(*tp == '.') ltp = tp;
			}
			// User might add .py on end of module name to load.  Make it work anyway
			if(ltp && streq(ltp, ".py")) *ltp = 0; // only want module name part
			jm_pyfile = htp;
			pos += 1;
			jm_numpyargs = argc - pos + 1;
			jm_pyargs = new char *[jm_numpyargs];
			jm_pyargs[0] = jm_mstr(argv[0]);
			int base = pos;
			for(; pos < argc; pos++) {
				jm_pyargs[pos-base+1] = jm_mstr(argv[pos]);
			}
		} else if(streq(s, "-mpi")) {
			// already set 
			pos++;
		} else if(secondarg && streq(s, "-blocksize")) {
			jm_block_size_arg = atoi(s2);
			if(jm_block_size_arg <= 0) {
				printf("Sched: Illegal block size: -blocksize %s\n", s2);
				jm_sched_abort();
			}
			pos += 2;
		} else if(secondarg && streq(s, "-alloctime")) {
			// might also be passed as part of -py - see prescan
			int rc = sscanf(s2, "%" PRIu64, &jm_alloc_time);
			if(rc != 1) {
				printf("Sched: Expecting integer number of seconds, got %s", s2);
				jm_sched_abort();
			}
			printf("Sched: allocation time = %" PRIu64 "\n", jm_alloc_time);
			pos += 2;
		} else if(streq(s, "-blocks")) {
			jm_collect_blocks = true;
			pos++;
		} else if(streq(s, "-v")) {
			// verbose arg
			pos++;
		} else {
			pos++; /* skip over */
		}
	}
}

//! \brief Figure out the rank in a lump for the head node of a block
int jm_block2rank(int bid) {
	int lbid = jm_block_use[bid].lbid; // Lump local block id
	Lump *lp = jm_block_use[bid].lp;
	return lp->bic_rank[lbid];
	// return lbid * jmres_block.getnumnodes(); // mult by number of nodes to get rank in Lump
}

//! \brief get the intercomm to talk to block bid
MPI_Comm jm_block2intercomm(int bid) {
	int lbid = jm_block_use[bid].lbid; // Lump local block id
	Lump *lp = jm_block_use[bid].lp;
	return lp->block_link_comm[lbid];
	// return jm_block_use[bid].lp->InterComm();
}

// the collection of jobs from the jobfile or other source.
vector<jm_job *> *jm_jobtab = nullptr;  // used to map jobid -> job
vector<jm_job *> *jm_sjobtab = nullptr; // priority sorted version of jm_jobtab, may be smaller

//! \brief Number of jobs in queue
int jm_GetNumJobs() {
	if(!jm_jobtab) return 0;
	return jm_jobtab->size();
}

// Constructor for job
jm_job::jm_job() {
	if(!jm_jobtab) {
		jm_jobtab = new vector<jm_job *>;
	}
	jobid = jm_jobtab->size() + 1;
	jm_jobtab->push_back(this); // add to table
	name = new char[32];
	sprintf(name, "job%d", jobid);
	pgm = nullptr;
	argbuf = nullptr;
	argbufsize = 0;
	dir = new char[3];
	strcpy(dir, ".");
	jobfile = nullptr;
	logfile = nullptr;
	wrapcmd = nullptr;
	startcmd = nullptr;
	depcmd = nullptr;
	priority = 0;
	ranktab = nullptr;
	dictp = PyDict_New();
	complain = true;

	state = jm_constructing;
	jobcmd[0] = JM_CMD_JOB;
	jobcmd[1] = -1; // set by encode
	jobcmd[2] = jobid;
	jobcmd[3] = 0;
	completionstatus = -1;
	startTime = endTime=0;
	estTime = 0;
	nodememKb = jmres_block.getnodemem(); // assume it uses it all unless told otherwise
	minnoderanks = 0;
}
// Destructor for jm_job
jm_job::~jm_job() {
	delete[] pgm;
	delete[] argbuf;
	delete[] dir;
	delete[] name;
	delete[] jobfile;
	delete[] logfile;
	delete[] startcmd;
	delete[] wrapcmd;
	delete[] depcmd;
	delete[] ranktab;
	Py_DECREF(dictp);
}

// convert state to string for use in Python interface
const char *jm_job::getstatestr() {
	switch(state) {
	case jm_constructing:  return "constructing";
	case jm_waiting:       return "waiting";
	case jm_sentcmd:       return "sentcmd";
	case jm_sentargs:      return "sentargs";
	case jm_statusreq:     return "statusreq";
	case jm_done:          return "done";
	case jm_aborted:       return "aborted";
	default:               return "unknown";
	}
}

//
// Add to queue - change state to waiting
int jm_job::queue() {
	if(state == jm_waiting)
		return 0;
	if(state == jm_constructing) {
		// Should validate completeness of job
		state = jm_waiting;
		// moved to after map/rank:  encodebuf();
		return 0; // success 
	} else {
		printf("ATTEMPT to queue job %s in state %s!\n", name, getstatestr());
		return -1;  // failure
	}
}

void jm_job::setnodemem(int Kb) {
	nodememKb = Kb;
}

int jm_job::getnodemem() {
	printf("Getting job %s node mem=%ld\n", name, (long) nodememKb);
	return nodememKb;
}

std::time_t jm_job::getEndTime() {
	return endTime;
}
std::time_t jm_job::getStartTime() {
	return startTime;
}
std::time_t jm_job::getEstTime() {
	return estTime;
}
void jm_job::setEstTime(std::time_t t){
	estTime = t;
}
void jm_job::setpgm(const char *apgm) {
	if(pgm) delete[] pgm;
	pgm = jm_mstr(apgm);
}
const char *jm_job::getpgm() {
	return pgm;
}
void jm_job::addarg(const char *val) {
	args.push_back(jm_mstr(val));
}
int jm_job::numargs() {
	return args.size();
}
void jm_job::clearargs() {
	args.clear();
}
const char *jm_job::getarg(int i) {
	if(i < 0 || i >= (int)args.size())
		return nullptr;
	return args[i];
}
int jm_job::numenv() {
	return env.size();
}
const char *jm_job::getenv(int i) {
	if(i < 0 || i >= (int)env.size())
		return nullptr;
	return env[i];
}
void jm_job::addenv(const char *val) {
	env.push_back(jm_mstr(val));
}
void jm_job::clearenv() {
	env.clear();
}
void jm_job::setwd(const char *adir) {
	if(dir) delete[] dir;
	dir = jm_mstr(adir);
}
const char *jm_job::getwd() {
	return dir;
}

const char *jm_job::getname() {
	return name;
}
void jm_job::setjobfile(const char *fname) {
	delete [] jobfile;
	jobfile = jm_mstr(fname);
}
const char *jm_job::getjobfile() {
	return jobfile;
}
void jm_job::setlogfile(const char *fname) {
	delete [] logfile;
	logfile = jm_mstr(fname);
}
const char *jm_job::getlogfile() {
	return logfile;
}
void jm_job::setpriority(int p) {
	priority = p;
}
int jm_job::getpriority() {
	return priority;
}
void jm_job::setdict(PyObject *datap) {
	Py_XDECREF(dictp);
	dictp = datap;
	Py_XINCREF(dictp);
}
PyObject *jm_job::getdict() {
	Py_XINCREF(dictp);
	return dictp;
}

//! \brief Set the name of the function to call when a job completes
void jm_job::setwrapcmd(const char *cmd) {
	if(wrapcmd) delete[] wrapcmd;
	if(strchr(cmd, '.')) {
		// explicit module spec, just use as is.
		wrapcmd = jm_mstr(cmd);
	} else {
		// if module spec is missing then use current module for spec.
		wrapcmd = new char[strlen(cmd) + strlen(jm_loadmodname) + 2];
		sprintf(wrapcmd, "%s.%s", jm_loadmodname, cmd);
	}
}

//! \brief Get the name of the function to call when a job completes
const char *jm_job::getwrapcmd() {
	return wrapcmd;
}
//! \brief Set name of function to call when job is about to start
void jm_job::setstartcmd(const char *cmd) {
	if(startcmd) delete[] startcmd;
	if(strchr(cmd, '.')) {
		startcmd = jm_mstr(cmd);
	} else {
		// if module spec is missing then use current module for spec.
		startcmd = new char[strlen(cmd) + strlen(jm_loadmodname) + 2];
		sprintf(startcmd, "%s.%s", jm_loadmodname, cmd);
	}
	printf("Set startcmd for job %s to %s\n", name, startcmd);
}
const char *jm_job::getstartcmd() {
	return startcmd;
}

//! \brief Set name of function to call when checking dependencies
void jm_job::setdepcmd(const char *cmd) {
	if(depcmd) delete[] depcmd;
	if(strchr(cmd, '.')) {
		depcmd = jm_mstr(cmd);
	} else {
		// if module spec is missing then use current module for spec.
		depcmd = new char[strlen(cmd) + strlen(jm_loadmodname) + 2];
		sprintf(depcmd, "%s.%s", jm_loadmodname, cmd);
	}
	printf("Set depcmd for job %s to %s\n", name, depcmd);
}

const char *jm_job::getdepcmd() {
	return depcmd;
}

//! \brief add or update a resource requirement for a job
int jm_job::setres(const char *resnames, int anranks, int anthreads) {
	int rmask = jmres_block.getresmask(resnames);
	printf("jm_job::setres:  rmask = %d\n", rmask);
	if(rmask <= 0) return -1;
	resmask = rmask;
	nranks = anranks;
	nthreadsperrank = anthreads;
	ranktab = new int[nranks];
	return 0;
}

//
// Get count of resources used by job.
// Some resources (cpu) are threaded and we have to multiply use by nthreadsperrank.
// Others like (gpu) are not threaded
void jm_job::getrestab(int *counts) {
	int i;
	for(i = 0;  i < jmres_block.numres(); i++) {
		counts[i] = 0;
		unsigned m = 1u << i;
		if(resmask & m) {
			int dc = nranks;
			// some resources are threaded
			if(jmres_block.res_threaded_mask & m) 
				dc *= nthreadsperrank;
			counts[i] += dc;
		}
	}
}

//
// blkres[resid] says how much is in use.
// jmres_block.getblockrescnt(i) gets available resources
// FIXME:  should return a cost, not a bool.
//
bool jm_job::blockHasRoom(jm_block_use_t& use) {
	int tcount;

	if(use.isLocked()) return false;

	// we either need nranks for resources like gpu's or nranks*nthreadsperrank for cpu resources
	tcount = nranks * nthreadsperrank; // number of slots for threaded resources
	// printf("Trying to place %d slots with resmask=%d\n", count, resmask);
	for(int resid = jmres_block.numres(); --resid >= 0;) {
		unsigned m = 1u << resid;
		if(resmask & m) { // we use this resource
			// if the new demand + current use > total
			int demand = (jmres_block.res_threaded_mask & m) ? tcount : nranks; // new demand
			if(demand + use.rescnt[resid] > jmres_block.getblockrescnt(resid)) {
				// printf("Sched: insufficient slots in block\n");
				return false; // no room
			}
		}
	}
	if(complain) {
		printf("Sched:  Room for job %s in block, checking detailed slot assignment\n", name);
	}
	// could be some issues with threads
	// TODO: improve efficiency 
	if(assignSlots(use)) {
		deassignSlots(use);
		return true;
	} else {
		return false;
	}
}

// figure out where we are going to be running this job
// a blockslot is a combination of a node and a slot on a node
//
// blockHasRoom has already told us that there is sufficient room in the block
// for this job.   We want to make the assignment efficiently.
// Suppose we have 2 node blocks with 32 cores and 2 gpu's that we have
// attached to slots 0 and 16.    slot 0 is cpu|gpu and slot 1 is just cpu.
// Jobs:
// 1) "gpu" job that requires 4 ranks of  cpu|gpu.
// 2) "cpu" job that requires 28 ranks of cpu
// 3) "cpu" job that requires 2 ranks x15 threads
//
// Now suppose that we start job (2) first.   We should consume pure cpu slots
// before adding cpu|gpu slots.   If a few slots are already consumed we might
// be forced to pick them.
// What is the best way to put them in?  If we were really clever we could
// collect the set of individually compatible jobs and pick the best packing set of jobs.
//
// Two jobs are not compatible if they would exceed the memory requirements.
//
// A simple minded solution for the moment is to select the best fit resources
// using a cost function that penalizes wasting resources.   Adding a cpu|gpu
// slot to a job that only needs cpus is more expensive.
// 
// jmres_block_type jmres_block has the detailed info of slots and resources.
//	int bslot2resmask(int bslot) returns resource mask for the block slot
//
bool jm_job::assignSlotsSub(jm_block_use_t& use, bool exact) {
	int numnodes = jmres_block.getnumnodes();
	int nodeslots = jmres_block.getnodeslots();
	int maxnodememKb = jmres_block.getnodemem();
	int node, t, bt, slot, bslot;
	int rid; // rank id, have to assign nranks ids to slots
	// find boundary for thread groups.  Each rank carries nthreadsperrank threads.
	int chunk = 1;
	while(chunk < nthreadsperrank) chunk <<= 1;
	// look for exact resource match first
	rid = 0; // which rank of job are we assigning
	// look through nodes in block to satisfy.  Done if all satisfied or out of nodes
	for(node = 0; node < numnodes && rid < nranks; node++) {
		// if we touch this node, then memory has to work.
		if(nodememKb + use.nodememKb[node] > maxnodememKb)  {
			// TODO: Comment out debug printf
			printf("Sched: job %s - nodememKb=%ld, use=%ld, maxmem=%ld - insufficient memory\n", name,
				(long)nodememKb, (long)use.nodememKb[node], (long)maxnodememKb);
			continue;
		}

		// go through slots on node by chunk size
		for(slot = 0; slot < nodeslots && rid < nranks; slot += chunk) {
			bslot = node * nodeslots + slot;
			int chunkend = slot + chunk;
			// last group of slots may be short if chunk does not divide nodeslots
			int tstop = chunk; 
			if(chunkend > nodeslots)
				tstop -= (chunkend - nodeslots); // subtract overflow
			// see if this chunk has room for a thread group
			int fcnt = 0; // number of satisfied threads in group
			for(t = 0; t < tstop && fcnt < nthreadsperrank; t++) {
				bt = bslot + t; // indexing through slots in a chunk
				if(use.slots[bt]) continue; // already assigned
				// first slot in rank is full resource mask.   Remainder
				// are only threaded resources
				int rm = resmask;
				if(fcnt) rm &= jmres_block.res_threaded_mask;
				if(exact) {
					// first look is for an exact match.
					// This will skip over slots with cpu|gpu when looking for cpu
					if(jmres_block.bslot2resmask(bt) == rm) 
						fcnt++;
				} else {
					// In a second try we will take cpu|gpu for cpu
					if((jmres_block.bslot2resmask(bt) & rm) == rm) 
						fcnt++;
				}
			}
			if(fcnt < nthreadsperrank) {
				if(complain) {
					const char *x = exact ? "exact" : "subset";
					// in some places like cores reserved for core-isolation we will
					// get 0.   Some cores have 4 slots  gpu|cpu,cpu,cpu,cpu   
					// In these cases we will find three exact matches, but in our
					// runs we give priority to the GPU jobs, which should have grabbed these cores
					if(fcnt && !exact)
						printf("Sched: job %s only found %d out of %d in node%d/%d %s\n", name, fcnt, nthreadsperrank, node, slot, x);
				}
				continue; // no room for thread group in this chunk.
			}
			fcnt = 0;
			//
			// mark group in use
			//
			for(t = 0; t < tstop; t++) {
				bt = bslot + t;
				if(use.slots[bt]) continue; // already assigned
				// compute resources needed for this slot.   If not the first
				// in thread group, then only look for threaded resources.
				// first slot in rank is full resource mask.   Remainder
				// are only threaded resources
				int rm = resmask;
				if(fcnt) rm &= jmres_block.res_threaded_mask;
				if(exact) {
					// first look is for an exact match.
					// This will skip over slots with cpu|gpu when looking for cpu
					if(jmres_block.bslot2resmask(bt) == rm) 
						use.slots[bt] = jobid;
				} else {
					// In a second try we will take cpu|gpu for cpu
					if((jmres_block.bslot2resmask(bt) & rm) == rm) 
						use.slots[bt] = jobid;
				}
			}
			ranktab[rid++] = bslot; // where is rank assigned (maybe we should track all threads)
		}
	}
	if(rid != nranks) {
		if(complain) {
			printf("Sched: job %s could only place %d of %d ranks\n", name, rid, nranks);
		}
		// reset all block slots back to unused
		int bscnt = jmres_block.blockslots();
		for(int bslot = 0; bslot < bscnt; bslot++) {
			if(use.slots[bslot] == jobid) use.slots[bslot] = 0;
		}
		return false;
	}
	//
	// Success!
	// Also track memory consumption.
	// Record the memory use in each node that has an active slot.
	//
	for(node = 0; node < numnodes; node++) {
		bslot = node * nodeslots;
		// see if we dropped any ranks in this node
		for(slot = nodeslots; --slot >= 0;) {
			if(use.slots[bslot+slot] == jobid) break;
		}
		if(slot >= 0) { // we did, add memory use
			use.nodememKb[node] += nodememKb;
		}
	}
	return true;
}

bool jm_job::assignSlots(jm_block_use_t& use) {
	if(assignSlotsSub(use, true))      // look for exact resource match, cpu can't go in cpu|gpu slot.
		return true;
	return assignSlotsSub(use, false); // allow assignment of cpu to cpu|gpu
}

// deassign slots for job and release memory
void jm_job::deassignSlots(jm_block_use_t& use) {
	int bscnt = jmres_block.blockslots();
	int nodeslots = jmres_block.getnodeslots();
	int lastnodesub = -1;
	for(int bslot = 0; bslot < bscnt; bslot++) {
		if(use.slots[bslot] == jobid) {
			use.slots[bslot] = 0;
			int node = bslot / nodeslots;
			if(node != lastnodesub) {
				use.nodememKb[node] -= nodememKb; // remove job memory
				lastnodesub = node; // so we only do it once per node
			}
		}
	}
}

//
// Add or subtract the job's requirements
// from the block's resources.
//
void jm_job::adjustBlockUse(int dir, jm_block_use_t& use) {
	int basedc = dir * nranks;
	int tdc = basedc * nthreadsperrank;
	for(int resid = jmres_block.numres(); --resid >= 0;) {
		unsigned rm = 1u << resid;
		if(resmask & rm) { // we use this resource
			int dc = (jmres_block.res_threaded_mask & rm) ? tdc : basedc;
			use.rescnt[resid] += dc;
		}
	}
	if(dir > 0) {
		assignSlots(use);
	} else {
		deassignSlots(use);
	}
}

void jm_job::setname(const char *aname) {
	if(name) delete name;
	name = jm_mstr(aname);
}

// pack rank->slot data into a comma separated string
// that we can transmit
char *jm_job::packrank() {
	char msbuf[20];
	int len = 0;
	for(int i = 0; i < nranks; i++) {
		sprintf(msbuf, "%d", ranktab[i]);
		len += strlen(msbuf) + 1;
	}
	char *buf = new char[len];
	char *cp = buf;
	*cp = 0;
	// pack rank info in a string
	for(int i = 0; i < nranks; i++) {
		if(i != 0) *cp++ = ',';
		sprintf(cp, "%d", ranktab[i]);
		cp += strlen(cp);
	}
	if(cp - buf >= len) {
		printf("jm_job::packrank: bad alloc\n");
		MPI_Finalize();
		exit(20);
	}
	return buf;
}

//
// Encode info needed by jm_master to launch the job in a buffer
// for a single transmission
// 
bool jm_job::encodebuf() {
	int i, asize;
	char *buf, *cp;
	char *rankstr;
	if(argbuf) delete argbuf;
	argbuf = nullptr;
	if(!pgm) {
		printf("job %s has no program name - skipping\n", name);
		state = jm_aborted;
		return false;
	}
	rankstr = packrank();
	if(!dir) dir = jm_mstr(".");
	// compute required size 
	asize = 0;
	asize += strlen(name) + JM_BUF_JOBNAME_LEN + 1;
	asize += strlen(pgm) + JM_BUF_PGM_LEN + 1;
	asize += strlen(dir) + JM_BUF_DIR_LEN + 1;
	if(logfile)
		asize += strlen(logfile) + JM_BUF_STDOUT_LEN + 1;
	for(i = 0; i < (int)args.size(); i++) {
		asize += strlen(args[i]) + JM_BUF_ARG_LEN + 1;
	}
	for(i = 0; i < (int)env.size(); i++) {
		asize += strlen(env[i]) + JM_BUF_ENV_LEN + 1;
	}
	asize += strlen(rankstr) + JM_BUF_RANK_LEN + 1;
	char nts[8];
	sprintf(nts, "%d", nthreadsperrank);
	asize += strlen(nts) + JM_BUF_NTHREAD_LEN + 1;
	asize += JM_BUF_END_LEN+1;

	buf = new char[asize + JM_XALLOC];
	cp = buf;

	/* name */
	strcpy(cp, JM_BUF_JOBNAME);
	cp += strlen(cp);
	strcpy(cp, name);
	cp += strlen(cp) + 1; /* skip over NUL */

	/* pgm */
	strcpy(cp, JM_BUF_PGM);
	cp += strlen(cp);
	strcpy(cp, pgm);
	cp += strlen(cp) + 1; /* skip over NUL */

	/* dir */
	strcpy(cp, JM_BUF_DIR);
	cp += strlen(cp);
	strcpy(cp, dir);
	cp += strlen(cp) + 1; /* skip over NUL */

	/* stdout */
	if(logfile) {
		strcpy(cp, JM_BUF_STDOUT);
		cp += strlen(cp);
		strcpy(cp, logfile);
		cp += strlen(cp) + 1; /* skip over NUL */
	}

	/* args */
	for(i = 0; i < (int)args.size(); i++) {
		strcpy(cp, JM_BUF_ARG);
		cp += strlen(cp);
		strcpy(cp, args[i]);
		cp += strlen(cp) + 1; /* skip over NUL */
	}

	/* env */
	for(i = 0; i < (int)env.size(); i++) {
		strcpy(cp, JM_BUF_ENV);
		cp += strlen(cp);
		strcpy(cp, env[i]);
		cp += strlen(cp) + 1; /* skip over NUL */
	}
	/* rank->slot data */
	strcpy(cp, JM_BUF_RANK);
	cp += strlen(cp);
	strcpy(cp, rankstr);
	cp += strlen(cp) + 1; /* skip over NUL too */
	delete[] rankstr;
	rankstr = nullptr;

	strcpy(cp, JM_BUF_NTHREAD);
	cp += strlen(cp);
	strcpy(cp, nts); // threads per rank
	cp += strlen(cp) + 1; /* skip over NUL too */

	strcpy(cp, JM_BUF_END);
	cp += strlen(cp) + 1;
	int nchars = (int)(cp - buf);
	if(nchars != asize) {
		printf("argbuf is wrong size: prediction %d, actual %d\n", asize, nchars);
	}
	argbufsize = asize;
	printf("Sched: argbufsize=%d\n", asize);
	jobcmd[1] = asize;
	argbuf = buf;
	return true;
}

//
// We start jobid's at 1 to avoid 0.   
//
jm_job *jmGetJobFromId(int jobid) {
	int jidx = jobid-1;
	if(jidx < 0 || jidx >= (int)jm_jobtab->size())
		return nullptr;
	return (*jm_jobtab)[jidx];
}

#define JM_EOFCH 1  // used as first character in EOF token

static FILE *jm_jobfp = nullptr;
static char *jm_linebuf = nullptr;
static int jm_linesize = 0;
static int jm_linelen = 0;
static int jm_linepos = 0;
static int jm_linenum = 0;
static bool jm_eof;
static int jm_ungetchar;
#define JM_EMPTY_CH (EOF-1)

static void jm_perr(const char *msg, ...) {
	char buf[2048];
	va_list args;
	va_start(args, msg);
	vsprintf(buf, msg, args);
	va_end(args);
	printf("Sched: Job File Error: %s\n", buf);
	if(jm_linebuf) {
		printf("Line %d: %s\n", jm_linenum, jm_linebuf);
	}
	jm_sched_abort();
}

void jm_setjobfp(FILE *fp) {
	jm_jobfp = fp;
	jm_linenum = 0;
	jm_eof = false;
	jm_linepos = jm_linelen; // trigger read on first getc
	jm_ungetchar = JM_EMPTY_CH;
}

void jm_readline() {
	int ch, pos;
	pos = 0;
	while(1) {
		ch = getc(jm_jobfp);
		if(ch == '\n' || ch == EOF) break;
		if(pos >= jm_linesize) {
			jm_linesize = jm_linesize ? jm_linesize*2 : 128;
			if(jm_linebuf) delete[] jm_linebuf;
			jm_linebuf = new char[jm_linesize + JM_XALLOC];
		}
		jm_linebuf[pos++] = ch;
	}
	if(pos == 0 && ch == EOF) jm_eof = true;
	if(pos >= jm_linesize) {
		jm_linesize = jm_linesize ? jm_linesize*2 : 128;
		if(jm_linebuf) delete[] jm_linebuf;
		jm_linebuf = new char[jm_linesize + JM_XALLOC];
	}
	jm_linebuf[pos] = 0;
	jm_linelen = pos;
	jm_linepos = 0; // for getc
	jm_linenum++;
}

int jm_getc() {
	int ch;

	if(jm_ungetchar != JM_EMPTY_CH) {
		ch = jm_ungetchar;
		jm_ungetchar = JM_EMPTY_CH;
		return ch;
	}
	if(jm_linepos >= jm_linelen) {
		jm_readline();
		return jm_eof ? EOF : '\n';
	}
	return jm_linebuf[jm_linepos++];
}

void jm_ungetc(int ch) {
	if(jm_ungetchar != JM_EMPTY_CH)
		jm_perr("Two stacked ungetch");
	jm_ungetchar = ch;
}

//
// When we exaust the job file, send a quit to all block rank 0 nodes.
// Then disconnect Finalize and exit.
//
void jm_quit() {
	MPI_Request *request;
	MPI_Status status;
	int flag;
	bool done[jm_block_count];
	int quitcmd[4];
	int i;

	printf("Sched: sending quit\n");
	quitcmd[0] = JM_CMD_QUIT;
	quitcmd[1] = 0;
	quitcmd[2] = 0;
	quitcmd[3] = 0;
	request = new MPI_Request[jm_block_count];
	// send everyone the message to quit 
	for(int bid = 0; bid < jm_block_count; bid++) {
		int blkrootrank = jm_block2rank(bid); 
		MPI_Isend(quitcmd, 4, MPI_INT, blkrootrank, JM_CMD_TAG, jm_block2intercomm(bid), &request[bid]);
		done[bid] = false;
	}
	// now poll until everyone has received the message
	while(1) {
		sleep(1);
		printf("Sched: sleeping during quit\n");
		for(i = jm_block_count; --i >= 0;) {
			if(done[i]) continue;
			MPI_Test(&request[i], &flag, &status);
			if(!flag) break; // still waiting
			done[i] = true;
		}
		if(i < 0) break; // all done
	}
	printf("Sched: all blocks have received quit\n");
#if 0
	// I have disabled the matching disconnect in jm_master.cc
	for(Lump *lp = jm_lumplist; lp; lp=lp->next)
		lp->Disconnect();
	printf("Sched: disconnect finished, attempt Finalize\n");
#else
	printf("Sched: disconnect disabled, attempt Finalize\n");
#endif
	fflush(stdout);

	MPI_Finalize();
	Py_Finalize();
	PyMem_RawFree(jm_pyprogram);
	printf("Sched: Finalize done\n");
	exit(0);
}

// returns new buffer.  Needs delete[]
static char *callStepFunc(jm_job *job, const char *cmd) {
	char *str;
	PyObject *tmp, *args[1];
	char *buf, *cp;
	const char *fname, *modname;

#if 0
	if(!cmd) {
		printf("Sched: calling step function <null>");
		return nullptr;
	} else {
		printf("Sched: calling step function %s\n", cmd);
	}
#endif
	// get handle for main module
	// set "job" variable to job that is completing
	tmp = (PyObject *)job->pyobj;
	Py_INCREF(tmp);
	args[0] = tmp;
	// Figure out module and function name
	buf = new char[strlen(cmd) + 1];
	strcpy(buf, cmd);
	cp = strchr(buf, '.');
	if(cp) {
		modname = buf;
		*cp = 0;
		fname = cp + 1;
	} else {
		modname = "__main__";
		fname = buf;
	}
	PyObject *pValue = callPythonFunc(modname, fname, 1, args);
	delete[] buf;
	Py_DECREF(tmp);
	// Now examine return value
	str = get_py_str(pValue); // returns new buffer
	if(pValue) {
		Py_XDECREF(pValue);
	}
	return str; // returns new buffer.  Needs delete[]
}


void jm_report_blocks() {
	if(jm_verbose < 1)
		return;
	int bid, node, slot, bslot;
	int ch;
	int numnodes = jmres_block.getnumnodes();
	int nodeslots = jmres_block.getnodeslots();
	int bslotcnt = numnodes * nodeslots;
	char *buf = new char[bslotcnt+1];
	buf[bslotcnt] = 0;

	printf("Sched: Block Slot Report:\n");
	for(bslot = 0; bslot < bslotcnt; bslot++) {
		switch(jmres_block.bslot2resmask(bslot)) {
		case 1: ch = 'C'; break;
		case 3: ch = 'G'; break;
		default: ch = 'U'; break;
		}
		buf[bslot] = ch;
	}
	printf("Sched: Block Def: %s\n", buf);
	for(bid = 0; bid < jm_block_count; bid++) {
		jm_block_use_t *bup = &jm_block_use[bid];
		for(node = 0; node < numnodes; node++) {
			for(slot = 0; slot < nodeslots; slot++) {
				bslot = node * nodeslots + slot;  // block level slot
				buf[bslot] = '0';
				if(bup->slots[bslot]) buf[bslot] = '1';
			}
		}
		printf("  Sched: B%d %s\n", bid, buf);
		printf("    Sched: ");
		for(int resid = jmres_block.numres(); --resid >= 0;) {
			printf("%d:%s ", bup->rescnt[resid], jmres_block.getresname(resid));
		}
		printf(" used \n");
	}
	printf("Sched: Block Slot Report Done\n");
	delete[] buf;
}

// return true if ordering a, b is good
int jm_job_priority_cmp(const void *avp, const void *bvp) {
	const jm_job *a = *(const jm_job **)avp;
	const jm_job *b = *(const jm_job **)bvp;
	// printf("Sched: comparing %s(p=%d) to %s(p=%d)\n", a->name, a->priority, b->name, b->priority);
	// large priority goes first
	if(a->priority != b->priority) {
		return (a->priority > b->priority) ? -1 : 1;
	}
	// otherwise schedule large runtime first
	if(a->estTime != b->estTime) {
		return a->estTime > b->estTime ? -1 : 1;
	}
	int at = a->nranks * a->nthreadsperrank;
	int bt = b->nranks * b->nthreadsperrank;
	return bt - at;
}

//
// Sort jobs by priority and size (proccnt)
// Should also schedule by estimated runtime.
// We want large jobs to launch first.
//
void jm_sortjobs() {
	printf("Sched: Sorting jobs\n");
	if(!jm_jobtab) {
		printf("Sched:  No job table to sort!\n");
		exit(1);
	}
	if(jm_sjobtab) delete jm_sjobtab;
	jm_sjobtab = new std::vector<jm_job *>(*jm_jobtab);
	size_t size = jm_sjobtab->size();
	jm_job **ja = jm_sjobtab->data();
	qsort((void *)ja, size, sizeof(jm_job *), jm_job_priority_cmp);
	printf("Sched: sorted table\n");
	for(size_t i = 0; i < size; i++) {
		// patch job id to match array
		jm_job *jp = ja[i];
		printf("Sched: %lu: %s, state=%s\n", (long unsigned)i, jp->getname(), jp->getstatestr());
	}
}

//
// Before starting, use
// job->blockHasRoom(jm_block_use[bid]) to test if the
// empty first block has room for the job.  If not,
// then declare the job unrunnable.
// TODO:  If we allow blocks to vary in size, then we
//        have to check all the "different" blocks
//        Some nodes may have more memory, or GPUs ...
static void jm_removeunrunnablejobs() {
	return;
	// Not supposed to touch jm_jobtab, will break jobid linkage.
	// We could remove from jm_sjobtab
	printf("Sched: Removing unrunnable jobs\n");
	size_t size = jm_jobtab->size();
	jm_job **ja = jm_jobtab->data();
	for(size_t i = 0; i < size; i++) {
		jm_job *jp = ja[i];
		if(!jp->blockHasRoom(jm_block_use[0])) {
			ja[i] = nullptr;
			printf("Sched: Removing job '%s'\n", jp->getname());
			delete jp;
		}
	}
	// now compress and change jobtab size
	size_t j = 0;
	for(size_t i = 0; i < size; i++) {
		if(ja[i]) ja[j++] = ja[i];
	}
	jm_jobtab->resize(j);
	printf("Sched: removed %lu unrunnable jobs\n", (unsigned long)(size - j));
}

//
// When we run out of jobs that are runnable, then we scan the jobs in
// the constructing state and see if any are runnable yet.
// The depcmd function will queue jobs if they are runnable.
//
int jm_check_job_dep() {
	int enablecnt = 0;
	int jobtabsize = jm_jobtab->size(); // jobs left to launch
	for(int i = 0; i < jobtabsize; i++) {
		jm_job *job = (*jm_jobtab)[i];
		if(job->state == jm_constructing) {
			// check if dependencies are satisfied yet
			const char *depcmd = job->getdepcmd();
			if(!depcmd) continue;

			// if null, not ready to go
			// msg: report
			if(job->state != jm_constructing) {
				printf("Sched: OOPS: job %s state changed to %s!\n", job->name, job->getstatestr());
			} else {
				char *str = callStepFunc(job, depcmd);
				if(!str) continue;
				if(*str) {
					// if msg, gives reason why not enabled.
					printf("Dep check enabled %s(%d): msg=%s\n", job->name, job->jobid, str);
				} else {
					enablecnt++;
				}
				delete[] str;
			}
		}
	}
	fflush(stdout);
	return enablecnt;
}

void jm_summary_report_jobs() {
	int constructing = 0; // python interface working on it.
	int waiting = 0;    // ready to run
	int sentcmd = 0;    // informed jm_master that job is coming
	int sentargs = 0;   // job arguments transmitted
	int statusreq = 0;  // job running, waiting for status
	int done = 0;       // job completed, could split into success and failure
	int aborted = 0;    // number of jobs aborted - startcmd says to abort
	int unknown = 0;
	jm_job *job;
	time_t t;
	time(&t);

	int jobtabsize = jm_jobtab->size(); // jobs left to launch
	for(int i = 0; i < jobtabsize; i++) {
		job = (*jm_jobtab)[i];
		switch(job->state) {
		case jm_constructing: constructing++; break;
		case jm_waiting: waiting++; break;
		case jm_sentcmd: sentcmd++; break;
		case jm_sentargs: sentargs++; break;
		case jm_statusreq: statusreq++; break;
		case jm_done:      done++;  break;
		case jm_aborted:   aborted++; break;
		default:           unknown++; break;
		}
	}
	char *tstr = jm_mstr(ctime(&t));
	char *cp = strrchr(tstr, '\n');
	if(cp) *cp = 0; // remove trailing newline
	printf("Sched: Report @%s: constructing=%d, ready=%d, scmd=%d, sargs=%d, running=%d, done=%d, abort=%d, unk=%d\n",
		tstr, constructing, waiting, sentcmd, sentargs, statusreq, done, aborted, unknown);
	delete[] tstr;
	fflush(stdout);
}

//
// Start working all jobs through their states
// waiting -> sentcmd -> setargs -> status request -> done
// Each of these is managed with a sequence of asyncronous
// send / receive requests.
//
void jm_run() {
	jm_removeunrunnablejobs();
	printf("Sched: Starting to run jobs\n");

	// Send welcome message to blocks
	for(int i = 0; i < jm_block_count; i++) {
		int blkrootrank = jm_block2rank(i);
		MPI_Comm intercomm = jm_block2intercomm(i);
		char buf[JM_WELCOME_SIZE];
		sprintf(buf, "block%d", i);
		MPI_Send(buf, JM_WELCOME_SIZE, MPI_CHAR, blkrootrank, 0, intercomm); 
		jm_log(0, "Sent welcome '%s' to block %d\n", buf, i);
	}

	if(!jm_jobtab) {
		printf("No jobs in job queue!\n");
		jm_quit();
	}
	jm_job *job;
	int bid;
	int jobtabsize = jm_jobtab->size(); // jobs left to launch
	jm_log(2, "Job tab size=%d\n", jobtabsize);
	bool active = true;
	int flag;
	int blkrootrank;
	int i;
	char *str;
	using std::chrono::system_clock;
	system_clock::time_point tpt;
	int passcnt = 0;
	int kid;
	int tick = 0;

	jm_sortjobs();
	bool didone = true;
	while(active) {
		if(!didone) {
			sleep(1);
			if(++tick > 10) {
				tick = 0;
				jm_summary_report_jobs();
			}
		}
		// printf("Sched: run pass\n");
		// see if any jobs need advancing
		active = false;
		// get time for this iteration
		tpt = system_clock::now();
		time_t tt = system_clock::to_time_t(tpt);
		didone = false;
		// first check all the waiting jobs to see if we can start some.
		for(i = 0; i < jobtabsize; i++) {
			job = (*jm_sjobtab)[i];
			if(job->state != jm_waiting) continue;
			active = true;
			// printf("Sched: found waiting job\n");
			// see if there is an open block
			bid = i % jm_block_count;
			for(kid = 0; kid < jm_block_count; kid++) {
				if(job->blockHasRoom(jm_block_use[bid])) break;
				if(++bid >= jm_block_count) bid = 0;
			}
			if(kid >= jm_block_count) {
				// printf("Sched: no nodes open for job %s\n", job->name);
			} else {
				if(job->complain)
					jm_log(2, "Sched: Resources available for job %s in block %d, trying to start it.\n", job->name, bid);
				// mark resources in use for block
				job->startTime = tt;
				str = callStepFunc(job, job->getstartcmd());
				if(str) {
					printf("Aborting job '%s', reason: %s\n", job->name, str);
					job->state = jm_aborted;
					delete [] str;
				} else {
					jm_block_use[bid].setLock(job->jobid);
					// reserve slots used by job
					job->adjustBlockUse(+1, jm_block_use[bid]);
					// encode pgm, cwd, env, args ...
					job->encodebuf();
					// jm_log(0, "Block %d is open for job '%s'\n", bid, job->name);
					// jm_log(2, "Sending %d:%d:%d:%d\n", job->jobcmd[0], job->jobcmd[1], job->jobcmd[2], job->jobcmd[3]);
					blkrootrank = jm_block2rank(bid);
					MPI_Comm intercomm = jm_block2intercomm(bid);
					MPI_Isend(job->jobcmd, 4, MPI_INT, blkrootrank,
					   JM_CMD_TAG, intercomm, &job->request);
					job->bid = bid;
					job->state = jm_sentcmd;
					// spawnwait = true; // HACK, spawn in progress
					didone = true;
				}
				jm_report_blocks();
			}
		}
		// now see if we can advance jobs along their state machines.
		for(i = 0; i < jobtabsize; i++) {
			job = (*jm_sjobtab)[i];
			switch(job->state) {
			case jm_done:
			case jm_aborted:
			case jm_constructing:
			case jm_waiting:
				continue;
			case jm_sentcmd:
				active = true;
				// printf("Sched: testing if job command received\n");
				MPI_Test(&job->request, &flag, &job->status);
				if(flag) { // command received
					// now send pgm+args
					// printf("Sched: job command has been received, sending arguments (%d bytes)\n", job->argbufsize);
					blkrootrank = jm_block2rank(job->bid);
					MPI_Comm intercomm = jm_block2intercomm(job->bid);
					MPI_Isend(job->argbuf, job->argbufsize, MPI_CHAR, blkrootrank,
					   JM_ARG_TAG, intercomm, &job->request);
					job->state = jm_sentargs;
					didone = true;
				}
				break;
			case jm_sentargs:
				active = true;
				// printf("Sched: testing if job arguments received\n");
				MPI_Test(&job->request, &flag, &job->status);
				if(flag) { // args received, now read status back
					blkrootrank = jm_block2rank(job->bid);
					MPI_Comm intercomm = jm_block2intercomm(job->bid);
					// printf("Sched: placing status receive request for job start\n");
					MPI_Irecv(&job->completionstatus, 1, MPI_INT, blkrootrank, 
						JM_STATUS_TAG + job->jobid, intercomm, &job->request);
					job->state = jm_statusreq;
					didone = true;
				}
				break;
			case jm_statusreq:
				active = true;
				// printf("Sched: testing job %s for completion status\n", job->name);
				MPI_Test(&job->request, &flag, &job->status);
				if(flag) {
					jm_log(0, "Sched: job %s receieved status from  B%d\n", job->getname(), job->bid);
					if(job->completionstatus == -1) { // tells us job started
						// once job has started we can allow another job to be launched
						// into the block.
						jm_block_use[job->bid].clrLock(job->jobid);
						jm_log(0, "Sched: job %s started on B%d\n", job->getname(), job->bid);
						blkrootrank = jm_block2rank(job->bid);
						MPI_Comm intercomm = jm_block2intercomm(job->bid);
						jm_log(0, "Sched: placing status receive request for job completion\n");
						MPI_Irecv(&job->completionstatus, 1, MPI_INT, blkrootrank, 
							JM_STATUS_TAG + job->jobid, intercomm, &job->request);
						// spawnwait = false;
					} else {
						// status received.   Job completed
						job->state = jm_done;
						job->endTime = tt;

						str = nullptr;
						if(job->completionstatus) {
							printf("Sched: Job %s failed with status %d\n", job->name, job->completionstatus);
						} else {
							str = callStepFunc(job, job->getwrapcmd());
							printf("Sched: Job %s wrapmsg='%s'\n", job->name, str ? str : "<no msg>");
						}
						// mark resources free again for block
						job->adjustBlockUse(-1, jm_block_use[job->bid]);
						jm_report_blocks();
						delete[] str;
					}
					didone = true;
				}
				break;
			}
		}
		passcnt++;
		if(!active || passcnt > 10) {
			passcnt = 0;
			int encnt = jm_check_job_dep();
			if(encnt) 
				active = true; // we found some new work
		}
	}
	printf("Sched: All jobs completed - quitting\n");
	jm_verify_block_mem0(); // make sure all memory was returned
	jm_quit();
}


/** Unix-like platform char * to wchar_t conversion. */
wchar_t *nstrws_convert(char *raw) {
  wchar_t *rtn = (wchar_t *) calloc(1, (sizeof(wchar_t) * (strlen(raw) + 1)));
  setlocale(LC_ALL,"en_US.UTF-8"); // Seriously!  Unless you do this python 3 crashes.
  mbstowcs(rtn, raw, strlen(raw));
  return rtn;
}

/** Dispose of an array of wchar_t * */
void nstrws_dispose(int count, wchar_t ** values) {
  for (int i = 0; i < count; i++) {
    free(values[i]);
  }
  free(values);
}

/** Convert an array of strings to wchar_t * all at once. */
wchar_t **nstrws_array(int argc, char *argv[]) {
  wchar_t **rtn = (wchar_t **) calloc(argc, sizeof(wchar_t *));
  for (int i = 0; i < argc; i++) {
    rtn[i] = nstrws_convert(argv[i]);
  }
  return rtn;
}

//
//! \brief Test if we were run from jm_master, which sets -mpi
//
static void jm_check_mpi(int argc, char *argv[]) {
	for(int i = 1; i < argc; i++) {
		if(streq(argv[i], "-mpi")) {
			jm_mpi_enable = true;
			break;
		}
	}
}

#if 0
//
// Machine parameters set from Python.
// Send them back to jm_master
void jm_send_machine_parameters() {
	if(!jm_mpi_enable)
		return;
	jm_block_count = jm_parent_size / jmres_block.getnumnodes();

	// Create tracking structure for blocks.
	jm_block_use = new jm_block_use_t[jm_block_count];

	char *slotenvbuf = jmres_block.getslotenvbuf();

	int mparms[JM_MACH_PARMS_SIZE]; // currently size 8
	mparms[0] = JM_MACH_PARMS_VERSION;
	mparms[1] = jmres_block.getnumnodes();    // number of nodes in block
	mparms[2] = jmres_block.getnodeslots();   // # of cpu slots
	mparms[3] = strlen(slotenvbuf) + 1;
	MPI_Bcast(mparms, JM_MACH_PARMS_SIZE, MPI_INT, MPI_ROOT, jm_sched_intercomm);
	printf("Sched: sending slotenvbuf: '%s'\n", slotenvbuf);
	MPI_Bcast(slotenvbuf, mparms[3], MPI_CHAR, MPI_ROOT, jm_sched_intercomm);
	delete[] slotenvbuf;
	printf("Sched: sent machine parameters\n");
}
#endif

void stophere() {
	printf("In stop here\n");
}

static void write_sched_env() {
	const char *sched_env_path = "sched_env.log";
	char cmdbuf[128];
	sprintf(cmdbuf, "env >& %s", sched_env_path);
	system(cmdbuf);
	printf("Sched: Wrote env to %s\n", sched_env_path);
}

//
// This is the main routine for the scheduler.   This
// single process collects the set of jobs and then
// feeds them to blocks.    Status is received asynchronously
// as jobs complete, freeing up the blocks for the next job.
// blocks and jobs have a resource mask, allowing jobs that
// don't overlap on resources to run in the same block.
//
// For testing, detect if run under MPI_Comm_spawn.  If not
// Then 
int main(int argc, char *argv[]
// On the Mac environ is not a global variable, but is passed as an arg to main
#ifdef __APPLE__
	, const char **environ
#endif
) { 
	MPI_Comm parent_intercomm; // connection to parent jm_masters if run that way
	int parent_size; // number of jm_master's in parent (number of nodes in parent)
	setlinebuf(stdout);
	write_sched_env(); // for debugging.
	printf("Sched: pid = %d\n", getpid());
	printf("Sched: parsing arguments\n");
	jm_sched_parseargs(argc, argv);

#if 1
	PyConfig config;
	PyConfig_InitPythonConfig(&config);
	PyConfig_SetString(&config, &config.program_name, jm_pyprogram);
	PyStatus astatus = PyConfig_SetBytesArgv(&config, jm_numpyargs, jm_pyargs);
	if(PyStatus_Exception(astatus) || jm_numpyargs < 1) {
		PyConfig_Clear(&config);
		printf("Argument list missing\n");
		exit(1);
	}
	jm_py_init(); // set up python mpi_jm module and types - call before Initialize
	jm_check_mpi(argc, argv);
	PyStatus status = Py_InitializeFromConfig(&config);
	if(PyStatus_Exception(status)) {
		PyConfig_Clear(&config);
		printf("Python initialization failure");
		exit(1);
	}
#else
	jm_pyprogram = Py_DecodeLocale(argv[0], nullptr);
	Py_SetProgramName(jm_pyprogram);
	jm_py_init(); // set up python mpi_jm module and types
	jm_check_mpi(argc, argv);
	Py_Initialize();

	// should change to $BINDIR/../pylib
	PyRun_SimpleString("import sys");
	//PyRun_SimpleString("sys.path.append(\".\")");

	// set up sys.argv to have the script name + following args
	printf("Sched: jm_numpyargs=%d\n", jm_numpyargs);
	for(int i = 0; i < jm_numpyargs; i++) {
		printf("Sched: jm_pyargs[%d] = '%s'\n", i, jm_pyargs[i]);
	}
	wchar_t **wargv = nstrws_array(jm_numpyargs, jm_pyargs);
	PySys_SetArgv(jm_numpyargs, wargv);
#endif
	if(jm_mpi_enable) {
		std::cout << "Sched: About to MPI_Init" << std::endl << std::flush;
		MPI_Init(&argc, &argv); 
		std::cout << "Sched: We initialized" << std::endl << std::flush;
		setlinebuf(stdout); // re-set.   MPI_Init messes with files
		jm_mpi_init_done = true;
		MPI_Comm_get_parent(&parent_intercomm); 
		std::cout << "Sched: We connected to parent" << std::endl;
		if(parent_intercomm != MPI_COMM_NULL) {
			std::cout << "Sched: trying remote size" << std::endl;
			MPI_Comm_remote_size(parent_intercomm, &parent_size); 
			std::cout << "Sched: success: parent size = " << parent_size << std::endl;
		} else {
			parent_size = 0;
            std::cout << "Sched: no intercommunicator :(" << std::endl;
		}
	} else {
		parent_size = 0;
	}
#if 0
	printf("Sched: Environment settings:\n");
	int ei = 0;
	while(environ[ei]) {
	   printf("Sched:    %s\n", environ[ei]);
	   ei++;
	}
#endif

	char procname[JM_LOC_ENT_SIZE];
	int procnamelen;
	if(jm_mpi_enable) {
		jm_log(2,"Sched: trying to get processor name\n");
		MPI_Get_processor_name(procname, &procnamelen);
		printf("Sched: starting up on '%s'\n", procname);
	} else {
		printf("Sched: Setting procname = dummy\n");
        strcpy(procname,"dummy");
    }
	sleep(1);   // increase if debugging for "gdb jm_sched <pid>"


	invokePythonScript("jm_machine", "setMachineParameters");
	if(parent_size) {
		// If started with mpirun -n <n> ... jm_master
		// Add parent as lump
		// Will build new comms to block rank0 members of jm_master
		jm_parent_lump = new Lump(parent_intercomm);
	}
	if(jm_collect_blocks) {
		// collect chunks of nodes until sent message that there aren't any more
		JmAcceptLumps();
	}
	jm_log(2, "we have read the machine parameters, now sending to master\n");
	JmSendMachineParameters(); // to all Lumps

	jm_log(2, "jm_master has received machine parms \n");
	// Call Python initial job collection
	// jm_pyfile
	if(!jm_pyfile) jm_pyfile = jm_mstr("jobs");
	invokePythonScript(jm_pyfile, "schedInit");

	jm_log("running jobs, %s\n", jm_lumplist ? "Have lump" : "No Lumps");
	if(jm_lumplist) {
		jm_run();              // distribute and monitor jobs
	} else {
		jm_log(0, "No parent to send jobs to.\n");
		if(jm_mpi_enable) {
			MPI_Finalize(); 
		}
		Py_Finalize();
		PyMem_RawFree(jm_pyprogram);
	}
	return 0; 
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
