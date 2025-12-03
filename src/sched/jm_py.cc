/*
 BSD 3-Clause License
 Copyright (c) 2025, The Regents of the University of California
 See Repository LICENSE file
 */
// This file implements a python3 interface to the shell
// model code.

// TODO:
// job priority
// runtime estimate
// support jobs that are flexible about resources.    
// Can run on 8/12/16/32 nodes with corresponding runtimes.
//   Need options / different executables for alternates.
// Add periodic call back

// Notes
// PyObject_TypeCheck(some_object, &MyType)  - verify type
//
// To reload a job file so we can execute it we will need
// importlib.reload(module)
// This will give us a clean module.
// May need
//    if 'myModule' in sys.modules:  
//    del sys.modules["myModule"]
//

#include "jm_sched.h"
#include "jm_mkdirp.h"

const char *jm_py_DocString = "Interface to MPI job manager";

// For debugging
void po(PyObject *p) {
    if(p) {
        PyObject_Print(p, stdout, 0);
        printf("\n");
    } else {
        printf("<NULL>\n");
    }
}


//
// Python has a convoluted notion of strings
// We just want something simple to to from a python unicode
// object to a char *.    There is a special case for file
// system paths.
//
// Return value should be deleted if not NULL
static const char *get_fs_str(PyObject *value) {
	PyObject *bytes;
	char *s;
	char *ns;
	Py_ssize_t len;

	if(!value || !PyUnicode_Check(value)) 
		return NULL;
	// http://chimera.labs.oreilly.com/books/1230000000393/ch15.html#_working_with_c_strings_of_dubious_encoding
	// bytes = PyUnicode_AsEncodedString(value,"utf-8","surrogateescape"); // generic
	bytes = PyUnicode_EncodeFSDefault(value); // special for file systems
	if (!bytes) return NULL;
	PyBytes_AsStringAndSize(bytes, &s, &len);
	ns = new char[strlen(s) + 1];
	strcpy(ns, s);
	Py_DECREF(bytes); // s may vanish now
	return(ns);
}

// Use generic encoding of unicode
// Return value must be deleted if not NULL
char *get_py_str(PyObject *value) {
	PyObject *bytes;
	char *s;
	char *ns;
	Py_ssize_t len;

	if(!value || !PyUnicode_Check(value)) 
		return NULL;
	// http://chimera.labs.oreilly.com/books/1230000000393/ch15.html#_working_with_c_strings_of_dubious_encoding
	bytes = PyUnicode_AsEncodedString(value,"utf-8","surrogateescape"); // generic
	if (!bytes) return NULL;
	PyBytes_AsStringAndSize(bytes, &s, &len);
	ns = new char[strlen(s) + 1];
	strcpy(ns, s);
	Py_DECREF(bytes); // s may vanish now
	return(ns);
}
// static PyObject *KsmError;

//
// Iterator for jm_job
// http://stackoverflow.com/questions/1815812/how-to-create-a-generator-iterator-with-the-python-c-api
//  The above example is Python 2.    Have to make some adjustments but
//  is good example.
//
typedef struct {
	PyObject_HEAD  // for python to manage
	int i;         // position in job table
} jm_job_iter;

// iter method always returns itself.  support: for ... in ...
PyObject *jm_job_iter_iter(PyObject *self) {
	Py_INCREF(self);
	jm_job_iter *p = (jm_job_iter *)self;
	p->i = 1;
	return self;
}

// how to get to next job
// jobids go from 1 .. NumJobs
PyObject *jm_job_iter_iternext(PyObject *self) {
	jm_job_iter *p = (jm_job_iter *)self;
	if(p->i <= jm_GetNumJobs()) {
		jm_job *jp = jmGetJobFromId(p->i);
		(p->i++);
		PyObject *tmp = (PyObject *)jp->pyobj;
		Py_INCREF(tmp);  // Check reference counting understanding
		return tmp;
	} else {
		/* Raising of standard StopIteration */
		PyErr_SetNone(PyExc_StopIteration);
		return NULL;
	}
}

static PyTypeObject jm_job_itertype = {
    PyObject_HEAD_INIT(0)
    "jm_job._jm_job_iter",     /*tp_name*/
    sizeof(jm_job_iter),       /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "Internal jm_job iterator object.",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    jm_job_iter_iter,  /* tp_iter: __iter__() method */
    jm_job_iter_iternext  /* tp_iternext: next() method */
};

//
// Now we need a "jobs" constructor for the iterator
//
static PyObject *jm_job_makeiter(PyObject *self, PyObject *args) {
	jm_job_iter *itp = PyObject_New(jm_job_iter, &jm_job_itertype);
	if(!itp) return NULL;
	if(!PyObject_Init((PyObject *)itp, &jm_job_itertype)) {
		Py_DECREF(itp);
		return NULL;
	}
	itp->i = 0; // start with first job
	return (PyObject *)itp;
}

// https://docs.python.org/3/extending/newtypes.html
typedef struct {
	PyObject_HEAD  // for python to manage
	int jobx;
	int junk;
} mpi_jm_job;

static void mpi_jm_job_dealloc(mpi_jm_job *self) {
	// chance here to free fields if needed
	Py_TYPE(self)->tp_free((PyObject *)self);
}

//
// Create a new job object
// Eventually will also create an underlying "real" job object
//
static PyObject *mpi_jm_job_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
	mpi_jm_job *self;
	jm_job *jp;
	PyObject *p;

	self = (mpi_jm_job *)type->tp_alloc(type, 0);
	jp = new jm_job(); // defaults to CPU job
	// assign unique ids to jobs
	// We will use these id's to look up the real job objects
	// This enables C code to delete real job objects without
	// worrying that Python still has a pointer.
	self->jobx = jp->jobid;
	printf("Created new job with id %d\n", jp->jobid);
	p = (PyObject *)self;
	Py_INCREF(p);  // job table maintains reference to PyObject
	jp->pyobj = p;
	jp->pyobj2 = p; // backup
	return p;
}

static int mpi_jm_job_init(mpi_jm_job *self, PyObject *args, PyObject *kwds)
{
	return 0;
}

static PyMemberDef mpi_jm_job_members[] = {
	{(char *)"junk", T_INT, offsetof(mpi_jm_job, junk), 0, (char *)"junk test field"},
	{NULL} /* Sentinel */
};

static PyObject *mpi_jm_job_jobx(mpi_jm_job *self) {
	return PyLong_FromLong(self->jobx);
}

#define GET_JM_JOB(jp) \
	jm_job *jp = jmGetJobFromId(self->jobx); \
	if(!jp) return NULL
#define GET_JM_JOBI(jp) \
	jm_job *jp = jmGetJobFromId(self->jobx); \
	if(!jp) return -1


static PyObject *mpi_jm_job_addarg(mpi_jm_job *self, PyObject *args) {
	const char *as;
	GET_JM_JOB(jp);

	if(!PyArg_ParseTuple(args, "s", &as)) return NULL;
	jp->addarg(as);
	Py_RETURN_NONE;
}

static PyObject *mpi_jm_job_addenv(mpi_jm_job *self, PyObject *args) {
	const char *name, *value;
	GET_JM_JOB(jp);

	if(!PyArg_ParseTuple(args, "ss", &name, &value)) return NULL;
	// Join strings with =
	char *buf = new char[strlen(name) + strlen(value) + 2];
	sprintf(buf, "%s=%s", name, value);
	jp->addenv(buf);
	delete[] buf;
	Py_RETURN_NONE;
}

// Get list of arguments for a job
static PyObject *mpi_jm_job_getargs(mpi_jm_job *self) {
	GET_JM_JOB(jp);

	int numargs = jp->numargs();
	PyObject *list = PyList_New(numargs);
	for(int i = 0; i < numargs; i++) {
		PyList_SetItem(list, i, Py_BuildValue("s", jp->getarg(i)));
	}
	return list;
}
static PyObject *mpi_jm_job_getenv(mpi_jm_job *self) {
	GET_JM_JOB(jp);

	int numenv = jp->numenv();
	PyObject *dict = PyDict_New();
	for(int i = 0; i < numenv; i++) {
		const char *s = jp->getenv(i);
		const char *cp = strchr(s, '=');
		int nlen = cp - s;
		char *name = new char[nlen + 1];
		strncpy(name, s, nlen);
		name[nlen] = 0;
		PyDict_SetItemString(dict, name, Py_BuildValue("s", cp + 1));
		delete[] name;
	}
	return dict;
}


static PyObject *mpi_jm_job_clearargs(mpi_jm_job *self) {
	GET_JM_JOB(jp);

	jp->clearargs();
	Py_RETURN_NONE;
}

static PyObject *mpi_jm_job_clearenv(mpi_jm_job *self) {
	GET_JM_JOB(jp);

	jp->clearenv();
	Py_RETURN_NONE;
}

//
// Add resource requirement to job
//
static PyObject *mpi_jm_job_setresources(mpi_jm_job *self, PyObject *args) {
	GET_JM_JOB(jp);
	const char *resname;
	int nranks, nthreads;
	if(!PyArg_ParseTuple(args, "sii", &resname, &nranks, &nthreads)) return NULL;
	if(jp->setres(resname, nranks, nthreads) < 0) {
		PyErr_SetString(PyExc_RuntimeError, "Unknown resource name");
		return NULL;
	}
	Py_RETURN_NONE;
}

static PyObject *mpi_jm_job_getresources(mpi_jm_job *self) {
	GET_JM_JOB(jp);
	int numres = jmres_block.numres();
	int *counts = new int[numres];
	jp->getrestab(counts);
	// allocate list that size
	PyObject *dict = PyDict_New();
	for(int i = 0; i < numres; i++) {
		if(counts[i] > 0) {
			PyDict_SetItemString(dict, jmres_block.getresname(i), Py_BuildValue("i", counts[i]));
		}
	}
	delete[] counts;
	return dict;
}

static PyObject *mpi_jm_job_queue(mpi_jm_job *self) {
	GET_JM_JOB(jp);

	if(jp->queue() < 0) {
		const char *statestr = jp->getstatestr();
		char *buf = new char[128];
		sprintf(buf, "Can't queue job unless state==constructing, state is %s", statestr);
		PyErr_SetString(PyExc_RuntimeError, buf);
		delete[] buf;
		return NULL;
	}
	Py_RETURN_NONE;
}

//
// Having trouble with pathlib causing segfaults.   This is a
// Hack replacement for pathlib rename.   The program is expected
// to write its output to wdir/jobstem.logtmp.   On successful
// completion, the job is renamed to wdir/jobstem.log
//
static PyObject *mpi_jm_rename_logtmp(mpi_jm_job *self) {
	char *msg = nullptr;
	int rc = -1;
	struct stat st;
	GET_JM_JOB(jp);
	const char *wd = jp->getwd();
	const char *jobname = jp->getname();
	const char *cp = strrchr(jobname, '/');
	if(cp) jobname = cp; // get the stem only
	// get a big enough buffer
	size_t len = strlen(wd) + strlen(jobname) + 60;
	char *buf = new char[len];
	cp = strrchr(jobname, '.');
	if(!cp || !streq(cp, ".yaml")) {
		sprintf(buf, "Missing '.yaml' from end of jobname '%s'", jobname);
	} else {
		sprintf(buf, "%s/%s", wd, jobname);
		char *bcp = strrchr(buf, '.'); // we know there is a .yaml at the end
		strcpy(bcp, ".logtmp");  // replace with .logtmp
		size_t off = bcp - buf;
		rc = stat(buf, &st);
		if(rc < 0) {
			sprintf(buf, "Missing logtmp for job '%s', errno=%d", jobname, errno);
			msg = buf;
		} else if(!S_ISREG(st.st_mode)) {
			sprintf(buf, "Missing logtmp for job '%s', not regular file", jobname);
			msg = buf;
		} else {
			char *nbuf = new char[len];
			strcpy(nbuf, buf);
			strcpy(&nbuf[off], ".log");
			rc = rename(buf, nbuf);
			if(rc < 0) {
				sprintf(buf, "rename to '%s' failed, errno=%d", nbuf, errno);
				msg = buf;
			}
			delete[] nbuf;
		}
	}
	if(msg) {
		PyObject *r = Py_BuildValue("s", msg);
		delete[] msg;
		return r;
	}
	Py_RETURN_NONE;
}

static PyObject *mpi_jm_getpathsuf(mpi_jm_job *self, PyObject *args) {
	GET_JM_JOB(jp);
	const char *wdir = jp->getwd();
	const char *name = jp->getname();
	if(!wdir) {
		PyErr_SetString(PyExc_RuntimeError, "working directory not set");
		return NULL;
	}
	char *suf = nullptr;
	if(!PyArg_ParseTuple(args, "s", &suf)) {
		PyErr_SetString(PyExc_RuntimeError, "Expecting new suffix");
		return nullptr;
	}

	const char *stem = strrchr(name, '/');
	if(!stem) stem = name;
	else stem++;
	size_t wlen = strlen(wdir);
	size_t jlen = strlen(stem);
	size_t slen = strlen(suf);
	char *buf = new char[wlen + jlen + slen + 10];
	strcpy(buf, wdir);
	char *cp = buf + wlen;
	if(cp[-1] != '/') *cp++ = '/';
	strcpy(cp, stem);
	cp = strrchr(cp, '.');
	if(!cp || !streq(cp, ".yaml")) {
		PyErr_SetString(PyExc_RuntimeError, "job name does not end in .yaml");
		return nullptr;
	}
	if(suf[0] == '.') suf++;
	strcpy(cp + 1, suf);
	PyObject *r = Py_BuildValue("s", buf);
	buf[0] = 'X';
	delete[] buf;
	return r;
}

static PyMethodDef mpi_jm_job_methods[] = {
	{"jobx", (PyCFunction)mpi_jm_job_jobx, METH_NOARGS, "Return jobx, hash id for job"},
	{"addarg", (PyCFunction)mpi_jm_job_addarg, METH_VARARGS, "Add program argument"},
	{"clearargs", (PyCFunction)mpi_jm_job_clearargs, METH_NOARGS, "Reset argument list"},
	{"addenv", (PyCFunction)mpi_jm_job_addenv, METH_VARARGS, "Add program argument"},
	{"clearenv", (PyCFunction)mpi_jm_job_clearenv, METH_NOARGS, "Reset environment list"},
	{"getargs", (PyCFunction)mpi_jm_job_getargs, METH_NOARGS, "Return argument list"},
	{"getenv", (PyCFunction)mpi_jm_job_getenv, METH_NOARGS, "Return argument list"},
	{"setresources", (PyCFunction)mpi_jm_job_setresources, METH_VARARGS, "set resource type, nranks, nthreadsperrank"},
	{"getresources", (PyCFunction)mpi_jm_job_getresources, METH_NOARGS, "Get job resources"},
	{"queue", (PyCFunction)mpi_jm_job_queue, METH_NOARGS, "queue to be run"},
	{"rename_logtmp", (PyCFunction)mpi_jm_rename_logtmp, METH_NOARGS, "rename wdir/jobname.logtmp to .log"},
	{"getpathsuf", (PyCFunction)mpi_jm_getpathsuf, METH_VARARGS, "generate path like logfile from wdir/<jobstem>"},
	{NULL} /* Sentinel */
};

//
// Working Directory
//
static PyObject *mpi_jm_job_get_wdir(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	return Py_BuildValue("s", jp->getwd());
}

static int mpi_jm_job_set_wdir(mpi_jm_job *self, PyObject *value, void *closer) {
	const char *s = get_fs_str(value); // dir is a file system string
	if(!s) {
        PyErr_SetString(PyExc_TypeError, "wdir must be a string");
        return -1;
	}
	GET_JM_JOBI(jp);
	jp->setwd(s);
	return 0;
}
//
// Memory requirement per node in Kb
//
static PyObject *mpi_jm_job_get_nodeMemKb(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	return Py_BuildValue("l", (long)jp->getnodemem());
}

static int mpi_jm_job_set_nodeMemKb(mpi_jm_job *self, PyObject *value, void *closer) {
	if(value && PyLong_Check(value)) {
		long v = PyLong_AsLong(value);
		GET_JM_JOBI(jp);
		printf("Setting job %s memory requirement to %ld\n", jp->getname(), (long)v);
		jp->setnodemem((int)v);
		return 0;
	} else {
		PyErr_SetString(PyExc_RuntimeError, "Memory use should be integer number of kB");
	}
	return -1;
}
//
// Program Name
//
static PyObject *mpi_jm_job_get_pgm(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	return Py_BuildValue("s", jp->getpgm());
}

static int mpi_jm_job_set_pgm(mpi_jm_job *self, PyObject *value, void *closer) {
	const char *s = get_fs_str(value); // dir is a file system string
	if(!s) {
        PyErr_SetString(PyExc_TypeError, "pgm path must be a string");
        return -1;
	}
	GET_JM_JOBI(jp);

	jp->setpgm(s);
	return 0;
}
//
// Job Name
//
static PyObject *mpi_jm_job_get_name(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	return Py_BuildValue("s", jp->getname());
}

static int mpi_jm_job_set_name(mpi_jm_job *self, PyObject *value, void *closer) {
	const char *s = get_py_str(value); // Simple str
	if(!s) {
        PyErr_SetString(PyExc_TypeError, "job name must be a string");
        return -1;
	}
	GET_JM_JOBI(jp);

	jp->setname(s);
	delete[] s;
	return 0;
}
//
// Job State
//
static PyObject *mpi_jm_job_get_state(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	return Py_BuildValue("s", jp->getstatestr());
}
static int mpi_jm_job_set_state(mpi_jm_job *self, PyObject *value, void *closer) {
	const char *s = get_py_str(value); // Simple str
	if(!s) {
        PyErr_SetString(PyExc_TypeError, "job name must be a string");
        return -1;
	}
	GET_JM_JOBI(jp);
	PyErr_SetString(PyExc_RuntimeError, "Can't set job state");
	delete[] s;
	return -1;
}

static PyObject *mpi_jm_job_getStartTime(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	long t = jp->getStartTime();
	return Py_BuildValue("l", t);
}

static int mpi_jm_job_setStartTime(mpi_jm_job *self, PyObject *value, void *closer) {
	PyErr_SetString(PyExc_RuntimeError, "Can't set job start time");
	return -1;
}

static PyObject *mpi_jm_job_getEndTime(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	long t = jp->getEndTime();
	return Py_BuildValue("l", t);
}

static int mpi_jm_job_setEndTime(mpi_jm_job *self, PyObject *value, void *closure) {
	PyErr_SetString(PyExc_RuntimeError, "Can't set job end time");
	return -1;
}

static PyObject *mpi_jm_job_getEstTime(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	long t = (long)jp->getEstTime();
	return Py_BuildValue("l", t);
}

static int mpi_jm_job_setEstTime(mpi_jm_job *self, PyObject *value, void *closure) {
	GET_JM_JOBI(jp);

	if(value && PyLong_Check(value)) {
		long v = PyLong_AsLong(value);
		jp->setEstTime((time_t)v);
		return 0;
	} else {
		PyErr_SetString(PyExc_RuntimeError, "Est Time value should be integer");
	}
	return -1;
}

// The jobfile is the path to the yaml file we got the job from
static PyObject *mpi_jm_job_getjobfile(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	const char *s = jp->getjobfile();
	if(!s) Py_RETURN_NONE;
	return Py_BuildValue("s", s);
}

static int mpi_jm_job_setjobfile(mpi_jm_job *self, PyObject *value, void *closer) {
	GET_JM_JOBI(jp);

	const char *ws = get_py_str(value);
	if(!ws) return -1;
	jp->setjobfile(ws);
	delete[] ws;
	return 0;
}

// The log file is where we want to send the stdout/stderr of the job
static PyObject *mpi_jm_job_getlogfile(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	const char *s = jp->getlogfile();
	if(!s) Py_RETURN_NONE;
	return Py_BuildValue("s", s);
}

static int mpi_jm_job_setlogfile(mpi_jm_job *self, PyObject *value, void *closer) {
	GET_JM_JOBI(jp);

	const char *ws = get_py_str(value);
	if(!ws) return -1;
	jp->setlogfile(ws);
	delete[] ws;
	return 0;
}

static PyObject *mpi_jm_job_getwrapcmd(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	const char *s = jp->getwrapcmd();
	if(!s) Py_RETURN_NONE;
	return Py_BuildValue("s", s);
}

static int mpi_jm_job_setwrapcmd(mpi_jm_job *self, PyObject *value, void *closer) {
	GET_JM_JOBI(jp);

	const char *ws = get_py_str(value);
	if(!ws) return -1;
	jp->setwrapcmd(ws);
	delete[] ws;
	return 0;
}

static PyObject *mpi_jm_job_getstartcmd(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	const char *s = jp->getstartcmd();
	if(!s) Py_RETURN_NONE;
	return Py_BuildValue("s", s);
}

static int mpi_jm_job_setstartcmd(mpi_jm_job *self, PyObject *value, void *closure) {
	GET_JM_JOBI(jp);

	const char *ws = get_py_str(value);
	if(!ws) return -1;
	jp->setstartcmd(ws);
	delete[] ws;
	return 0;
}

static PyObject *mpi_jm_job_getdepcmd(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	const char *s = jp->getdepcmd();
	if(!s) Py_RETURN_NONE;
	return Py_BuildValue("s", s);
}

static int mpi_jm_job_setdepcmd(mpi_jm_job *self, PyObject *value, void *closer) {
	GET_JM_JOBI(jp);

	const char *ws = get_py_str(value);
	if(!ws) return -1;
	jp->setdepcmd(ws);
	delete[] ws;
	return 0;
}

static int mpi_jm_job_set_priority(mpi_jm_job *self, PyObject *value, void *closure) {
	GET_JM_JOBI(jp);
	if(value && PyLong_Check(value)) {
		long v = PyLong_AsLong(value);
		jp->setpriority((int)v);
		return 0;
	} else {
		PyErr_SetString(PyExc_RuntimeError, "job priority value should be integer");
		return -1;
	}
}

static PyObject *mpi_jm_job_get_priority(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);
	return Py_BuildValue("i", jp->getpriority());
}

static int mpi_jm_job_setdict(mpi_jm_job *self, PyObject *value, void *closure) {
	GET_JM_JOBI(jp);
	if(value && PyDict_Check(value)) {
		jp->setdict(value);
		return 0;
	} else {
		PyErr_SetString(PyExc_RuntimeError, "job dict field should be dictionary");
		return -1;
	}
}

static PyObject *mpi_jm_job_getdict(mpi_jm_job *self, void *closure) {
	GET_JM_JOB(jp);

	return jp->getdict();  // does INCREF
}

static PyGetSetDef mpi_jm_job_getseters[] = {
	{(char *)"nodeMemKb", (getter) mpi_jm_job_get_nodeMemKb, (setter)mpi_jm_job_set_nodeMemKb, (char *)"memory per node(Kb)", NULL},
	{(char *)"wdir", (getter) mpi_jm_job_get_wdir, (setter)mpi_jm_job_set_wdir, (char *)"working directory", NULL},
	{(char *)"pgm", (getter) mpi_jm_job_get_pgm, (setter)mpi_jm_job_set_pgm, (char *)"executable path", NULL},
	{(char *)"name", (getter) mpi_jm_job_get_name, (setter)mpi_jm_job_set_name, (char *)"name of job", NULL},
	{(char *)"priority", (getter) mpi_jm_job_get_priority, (setter)mpi_jm_job_set_priority, (char *)"priority of job"},
	{(char *)"state", (getter) mpi_jm_job_get_state, (setter)mpi_jm_job_set_state, (char *)"exec state of job", NULL},
	{(char *)"startTime", (getter) mpi_jm_job_getStartTime, (setter)mpi_jm_job_setStartTime, (char *)"start time of job", NULL},
	{(char *)"endTime", (getter) mpi_jm_job_getEndTime, (setter)mpi_jm_job_setEndTime, (char *)"end time of job", NULL},
	{(char *)"estTime", (getter) mpi_jm_job_getEstTime, (setter)mpi_jm_job_setEstTime, (char *)"estimated run time of job", NULL},
	{(char *)"jobfile", (getter) mpi_jm_job_getjobfile, (setter)mpi_jm_job_setjobfile, (char *)"file with startup/wrapup code", NULL},
	{(char *)"logfile", (getter) mpi_jm_job_getlogfile, (setter)mpi_jm_job_setlogfile, (char *)"file with startup/wrapup code", NULL},
	{(char *)"startcmd", (getter) mpi_jm_job_getstartcmd, (setter)mpi_jm_job_setstartcmd, (char *)"cmd with startup code", NULL},
	{(char *)"wrapcmd", (getter) mpi_jm_job_getwrapcmd, (setter)mpi_jm_job_setwrapcmd, (char *)"cmd with wrapup code", NULL},
	{(char *)"depcmd", (getter) mpi_jm_job_getdepcmd, (setter)mpi_jm_job_setdepcmd, (char *)"function to check dependencies before queuing", NULL},
	{(char *)"dict", (getter) mpi_jm_job_getdict, (setter)mpi_jm_job_setdict, (char *)"dictionary to describe job", NULL},
	{NULL} /* Sentinel */
};

//
// Export of type for jm_job to python
//
static PyTypeObject mpi_jm_job_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"mpi_jm.job",			   /* tp_name */
	sizeof(mpi_jm_job),        /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)mpi_jm_job_dealloc,        /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
	   Py_TPFLAGS_BASETYPE,    /* tp_flags */
    "mpi_jm Job objects",      /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    mpi_jm_job_methods,        /* tp_methods */
    mpi_jm_job_members,        /* tp_members */
    mpi_jm_job_getseters,      /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)mpi_jm_job_init, /* tp_init */
    0,                         /* tp_alloc */
    mpi_jm_job_new,            /* tp_new */
};

//
// Exec "system" command
// Run command under shell
//
static PyObject *jm_py_System(PyObject *self, PyObject *args) {
	const char *command = nullptr;
	int sts;

	if (!PyArg_ParseTuple(args, "s", &command)) return NULL;
	sts = system(command);
	return PyLong_FromLong(sts);
}

//
// Used to avoid crashes in pathlib.mkdir on Summit  (ppc64le)
// 
static PyObject *jm_py_mkdirp(PyObject *self, PyObject *args) {
	const char *dir = nullptr;
	if (!PyArg_ParseTuple(args, "s", &dir)) return NULL;
	int rc = jm_mkdirp(dir);
	if(rc < 0) {
		const char *estr = strerror(errno);
		if(!estr) estr = "<unknown error>";
		return PyUnicode_FromString(estr);
	}
	Py_RETURN_NONE; // success
}

//
// Add resource type and count to a node
//
static PyObject *jm_py_addnoderesource(PyObject *self, PyObject *args) {
	char *name;
	if(!PyArg_ParseTuple(args, "s", &name)) return NULL;
	jmres_block.addres(name);
	Py_RETURN_NONE;
}
static PyObject *jm_py_getnoderesources(PyObject *self, PyObject *args) {
	int num = jmres_block.numres();
	PyObject *list = PyList_New(num);
	for(int i = 0; i < num; i++) {
		PyList_SetItem(list, i, Py_BuildValue("s", jmres_block.getresname(i)));
	}
	return list;
}

static PyObject *jm_py_addslotenv(PyObject *self, PyObject *args) {
	char *name, *value;
	int slot;
	if(!PyArg_ParseTuple(args, "iss", &slot, &name, &value)) return NULL;
	jmres_block.addslotenv(slot, name, value);
	Py_RETURN_NONE;
}

//
// set number of nodes in a block
//
static PyObject *jm_py_setblocksize(PyObject *self, PyObject *args) {
	int count;

	if(!PyArg_ParseTuple(args, "i", &count)) return NULL;
	if(count < 1) {
        PyErr_SetString(PyExc_ValueError, "count must be positive");
		return NULL;
	}
	jmres_block.setnumnodes(count);
	Py_RETURN_NONE;
}

static PyObject *jm_py_getblocksize(PyObject *self) {
	return PyLong_FromLong(jm_block_size_arg);
}

static PyObject *jm_py_addslot(PyObject *self, PyObject *args) {
	char *res;
	if(!PyArg_ParseTuple(args, "s", &res)) return NULL;
	if(!res || !*res ) {
        PyErr_SetString(PyExc_ValueError, "empty resource list");
		return NULL;
	}
	jmres_block.addslot(res);
	Py_RETURN_NONE;
}

static PyObject *jm_py_resfinish(PyObject *self, PyObject *args) {
	jmres_block.finish();
	Py_RETURN_NONE;
}

static PyObject *jm_py_sortjobs(PyObject *self, PyObject *args) {
	jm_sortjobs();
	Py_RETURN_NONE;
}


static PyObject *jm_py_setnodemem(PyObject *self, PyObject *args) {
	long gb;
	if(!PyArg_ParseTuple(args, "l", &gb)) return NULL;
	if(gb <= 0) {
		PyErr_SetString(PyExc_ValueError, "node memory must be positive");
		return NULL;
	}
	jmres_block.setnodemem((size_t)gb);
	Py_RETURN_NONE;
}

static PyObject *jm_py_getnodemem(PyObject *self, PyObject *args) {
	return Py_BuildValue("l", (long)jmres_block.getnodemem());
}

//
// mpi_jm methods
//
static PyMethodDef mpi_jm_Methods[] = {
    {"system",  jm_py_System, METH_VARARGS, "Execute a shell command."},
	{"jobs", (PyCFunction)jm_job_makeiter, METH_NOARGS, "Return iterator for jobs"},
	{"addnoderesource", (PyCFunction)jm_py_addnoderesource, METH_VARARGS, "Define resources in node"},
	{"getnoderesources", (PyCFunction)jm_py_getnoderesources, METH_NOARGS, "Return resource type list"},
	{"addslotenv", (PyCFunction)jm_py_addslotenv, METH_VARARGS, "Define env for node slot"},
	{"setblocksize", (PyCFunction)jm_py_setblocksize, METH_VARARGS, "Sets number of nodes in block"},
	{"getblocksize", (PyCFunction)jm_py_getblocksize, METH_NOARGS, "Gets number of nodes in a block"},
	{"addslot", (PyCFunction)jm_py_addslot, METH_VARARGS, "Adds slot to resources"},
	{"setnodemem", (PyCFunction)jm_py_setnodemem, METH_VARARGS, "Sets memory(Kb) for node"},
	{"getnodemem", (PyCFunction)jm_py_getnodemem, METH_NOARGS, "Gets memory(Kb) for node"},
	{"resfinish", (PyCFunction)jm_py_resfinish, METH_NOARGS, "Totals resources"},
	{"sortjobs", (PyCFunction)jm_py_sortjobs, METH_NOARGS, "sorts jobs into priority order"},
	{"mkdirp", (PyCFunction)jm_py_mkdirp, METH_VARARGS, "Create directory path"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

//
// Descriptor for the mpi_jm Module
//
static struct PyModuleDef mpi_jm_Module = {
	PyModuleDef_HEAD_INIT,
	"mpi_jm", /* name of module */
	(char *)jm_py_DocString, /* module documentation or NULL */
	-1,               /* size of per-interpreter state of the module */
	                  /* or -1 if the module keeps state in global variables */
	mpi_jm_Methods
};

//
// Install the mpi_jm module
// add the "job" type to the module
// add iterator over jobs
//
static PyObject *PyInit_mpi_jm_mod() {
	PyObject *m;
	// set up job type
	mpi_jm_job_type.tp_new = mpi_jm_job_new;
	if(PyType_Ready(&mpi_jm_job_type) < 0)
		return NULL; // failed 
	jm_job_itertype.tp_new = PyType_GenericNew;
	if(PyType_Ready(&jm_job_itertype) < 0) return NULL;
	
	// create mpi_jm module
	m =  PyModule_Create(&mpi_jm_Module);
	if(m == NULL)
		return NULL; // failed
	
	// add job type to module
	Py_INCREF(&mpi_jm_job_type);
	PyModule_AddObject(m, "job", (PyObject *)&mpi_jm_job_type);

	Py_INCREF(&jm_job_itertype);
	PyModule_AddObject(m, "_jm_job_iter", (PyObject *)&jm_job_itertype);

	return m;
}

//
// Convert a PyObject to a jm_job (if of the correct type)
// 
jm_job *jm_pyjob2job(PyObject *obj) {
	if(PyObject_IsInstance(obj, (PyObject *)&mpi_jm_job_type)) {
		mpi_jm_job *self = (mpi_jm_job *)obj;
		GET_JM_JOB(jp);
		return jp;
	} else {
		printf("Sched: jm_pyjob2job: pyobj is wrong type\n");
		return nullptr;
	}
}

int jm_pyjob2jobx(PyObject *obj) {
	if(PyObject_IsInstance(obj, (PyObject *)&mpi_jm_job_type)) {
		mpi_jm_job *self = (mpi_jm_job *)obj;
		return self->jobx;
	} else {
		return -1;
	}
}

// After sorting, we update indices.
void jm_patchjobid(jm_job *jp, int id) {
	mpi_jm_job *self = (mpi_jm_job *)jp->pyobj;
	self->jobx = id;
	jp->jobid = id;
}

void jm_py_init() {
	// Set up for automatic load into Python interpreter
	PyImport_AppendInittab("mpi_jm", &PyInit_mpi_jm_mod);
}
