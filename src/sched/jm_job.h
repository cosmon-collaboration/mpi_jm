

// Lock around spawn at the Lump level instead of the block level
#define LUMPSPAWNLOCK

class Lump;

// states a job can be in
enum jm_jobstate {
	jm_constructing,  // under construction skip over until waiting
	jm_waiting,   // nothing happening with job yet
	jm_sentcmd,   // sent initial job request
	jm_sentargs,  // sent arguments, Irecv active for status
	jm_statusreq, // waiting for status
	jm_aborted,   // job aborted - just before starting
	jm_done       // job completed - could log/free
	};

// keeps track of all slots in use
class jm_block_use_t {
#ifndef LUMPSPAWNLOCK
private:
	int lockval; // set to jobid when held.  0 when not held.
#endif
public:
	// rescnt is used as a first filter.
	int *rescnt; // usage by resource type
	int *slots;   // store owning job id
	int *nodememKb;  // used node mem
	Lump *lp;    // Lump this belongs to.
	int  lbid;   // block id in Lump - gives rank in Lump

	jm_block_use_t();
	~jm_block_use_t();
	void doublelock();
	void badunlock(int jobid);
	void setLock(int jobid);
	bool isLocked();
	void clrLock(int jobid);
	int blkUsedMemKb();
};

//
// Lumps are collections of nodes running jm_master that connect
// via a published name and accept/connect to offer their services
//
class Lump {
#ifdef LUMPSPAWNLOCK
public:
	int lockval; // set to jobid when held.  0 when not held
#endif
private:
	bool parentlump;
	char *lumpname;
	int lumpsize;
	int block_count;
	char **block_link_str;  // from MPI_Open_port for each block
	MPI_Comm *block_link_comm; //resulting comm from str
	int *bic_rank; // intercommunicator rank to use to reach block
public:
	Lump *next; // for looping over Lumps
public:
	void CreateMsg() {
		printf("Created Lump %s with %d nodes\n", lumpname, lumpsize);
	}
	Lump(char *alumpname, MPI_Comm tmpcomm); // for accept/connect lumps
	Lump(MPI_Comm aintercomm); // Used to track Lump created in initial launch of jm_master
	const char *Name() { return lumpname; }

#ifdef LUMPSPAWNLOCK
	// We move locking to the lump level
	void doublelock();
	void badunlock(int jobid);
	inline void setLock(int jobid) { if(lockval) doublelock(); lockval = jobid; }
	inline bool isLocked() { return lockval != 0; }
	inline void clrLock(int jobid) { if(lockval != jobid) badunlock(jobid); lockval = 0;}
#endif

	//! \brief Get number of blocks in Lump.
	int BlockCount() { return block_count; }
	MPI_Comm InterComm(int bid) { return block_link_comm[bid]; }
	void SendMachineParameters();
	void Disconnect();
	void MakeBlockLinks(MPI_Comm tmpcomm);

	friend int jm_block2rank(int bid);
	friend MPI_Comm jm_block2intercomm(int bid);
};

//
// Starting work on configurations
// One job can have multiple configurations.
// Configurations can be shared between jobs
// j.addconfig(cfg) can be used to add the configuration to the job.
//
#if 0
#define JM_MAX_RES_CHT
struct jm_slot { // trying to pack in 16 bits
	// start of node.   A pattern can use multiple nodes
	// nextnode says to skip to a new node.   Will be set for first slot.
	// Do we start counting at the node slot 0 or at a cpu slot 0.
	// The
	unsigned align : 2;       // set to snap to boundary
	unsigned pack : 2;        // set to stay in group 
#define JM_NONE 0
#define JM_NODE 1
#define JM_CPU  2
#define JM_CORE 3
	// Example align=CORE, pack=CPU means stay in same CPU and get a new core
	//         align=NONE, pack=CORE means get another slot in same CORE
	unsigned resmask : JM_MAX_RES_CNT; // resources types needed for slot
};
// We reserve CPU slots
struct jm_reserve {
	unsigned node : 20;       // node number in supercomputer
	unsigned slot : 12;       // which slot (thread)
	unsigned_t  resmask  : JM_MAX_RES_CNT;       // other resource reservations by bitmask
							  // GPU0, GPU1, GPU2, GPU3, GPU5, GPU6, ...
};
struct jm_config {
	jm_config *next;          // A job can have alternate configurations.
	std::time_t  estTime;     // estimated runtime for this configuration
	unsigned nodememMb;       // required memory on node in Mb

	uint32_t numnodes;        // total number of nodes required for job
	// summary for fast checks
	uint8_t  resperpat[JM_MAX_RES_CNT];  // number of resources per pattern of each type
	uint16_t nodesperpat;     // nodes per pattern, divides numnodes to give numpattern
	uint16_t threadsperrank;  // number of threads associated with ranks
	// patterns can cross nodes
	uint16_t numslots;        // slots in pattern
	jm_slot *pattern;         // set of slots in a pattern corresponding to ranks

	// match against open resources
	uint32_t *Match();		  // returns global slot ids (placement), nullptr for failure
};
#endif

class jm_job {
public:
	int jobid;             // set to position in jobtab.  Starts with 1
	void *pyobj;
private:
	// vector <int> restable;   // pairs of resource id/count
	jm_jobstate state;       // what is going on with job
	int bid;                 // block executing job
	int jobcmd[4];           // used to send job start with async send - job private buffer
	MPI_Request request;     //  args for Isend
	MPI_Status status;       //  and for Irecv
	int completionstatus;    // run result.
	// packed data needed to start job
	int argbufsize;          // bytes in argbuf
	char *argbuf;            // pgm + args, NUL separated, double NUL at end
	// data to be packed into argbuf
	char *pgm;               // name of program to run
	vector <char *> args;    // program arguments
	vector <char *> env;     // table of environment variables
	char *dir;               // directory to run in
	char *name;              // job name
	unsigned  resmask;            // bitmask for resources in jmres_block
	int  nranks;             // number of ranks to launch
	int  nthreadsperrank;    // number of threads to reserve per rank
	int  minnoderanks;       // lower limit on number of ranks on a node
	int  nodememKb;          // memory per node in Gb.
	int *ranktab;

	// managment data
	char *jobfile;           // optional file holding code
	char *logfile;           // where to write stdout/stderr
	char *startcmd;          // command to exec at start
	char *wrapcmd;           // command to exec at end
	char *depcmd;            // function to check dependencies
	std::time_t   startTime; // Time job started
	std::time_t   endTime;   // Time job completed
	std::time_t   estTime;   // Estimated run time set outside
	int priority;            // Part of picking next job to run
						     // 0 normal, larger is higher
	PyObject *dictp;         // Used to store Python dictonary

	bool complain;           // set at beginning for messages

public:
	void *pyobj2;            // copy for checking
public:
	jm_job();
	~jm_job();
	int setres(const char *resnames, int nranks, int nthreads);
	const char *getstatestr();     // state as string for Python
	void setpgm(const char *apgm); // exec path
	const char *getpgm();          // read exec path
	void addarg(const char *val);  // add program argument
	int  numargs();
	int  numenv();
	const char *getarg(int i);
	void clearargs();	   // reset args array to empty
	void addenv(const char *val);  // add environment setting
	const char *getenv(int i);
	void clearenv();       // reset env to empty
	void setwd(const char *adir);  // set working dir for run
	const char *getwd();           // get current working dir
	void setname(const char *aname);  // name to call job
	const char *getname();         // read job name
	void setnodemem(int Kb);       // per node memory requirement
	int  getnodemem();
	// bool blockHasRoom(int *blkres); // test block for room to hold job
	bool blockHasRoom(jm_block_use_t& use); // test block for room to hold job
	void adjustBlockUse(int dir, jm_block_use_t& use); // inc (+1)  or dec (-1)  use
	bool assignSlotsSub(jm_block_use_t& use, bool exact);
	bool assignSlots(jm_block_use_t& use);
	void deassignSlots(jm_block_use_t& use);
	bool encodebuf();        // encodes job parameters for distribution to jm_master
	char *packrank();
	std::time_t getStartTime();
	std::time_t getEndTime();
	std::time_t getEstTime();
	void setEstTime(std::time_t t);
	void getrestab(int *counts); // return counts indexed by resid
	int queue();   // move status from constructing to waiting.  ret 0-success, -1 error
	void setjobfile(const char *fname);
	const char *getjobfile();
	void setlogfile(const char *fname);
	const char *getlogfile();
	void setwrapcmd(const char *cmd);
	const char *getwrapcmd();
	void setstartcmd(const char *cmd);
	const char *getstartcmd();
	void setdepcmd(const char *cmd); // check dependencies so to see if it can be queued.
	const char *getdepcmd();
	void setpriority(int p);
	int getpriority();
	void setdict(PyObject *);
	PyObject *getdict();
	friend int jm_check_job_dep();
	friend void jm_run();
	friend void jm_process_jobfile();
	friend int jm_job_priority_cmp(const void *, const void *);
	friend void jm_summary_report_jobs();
};

extern void jm_sortjobs();

// py interface uses id's that can be checked for validity
extern jm_job *jmGetJobFromId(int jobid);
extern int jm_GetNumJobs();
extern const char *jm_loadmodname;   // used for callbacks to add <mod>. to front of function names that don't have them.
extern void JmAcceptLumps();         // Listen for Lumps of nodes offering services
extern void JmSendMachineParameters(); // send machine parameters to all Lumps

extern Lump *jm_lumplist;            // chain of all registered lumps
extern int jm_block_count;           // number of blocks
extern jm_block_use_t *jm_block_use; // tracks slot usage for each block
