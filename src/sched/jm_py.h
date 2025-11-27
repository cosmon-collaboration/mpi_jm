#include <Python.h>
#include "structmember.h"
void jm_py_init();
char *get_py_str(PyObject *value);

int jm_pyjob2jobx(PyObject *obj);
jm_job *jm_pyjob2job(PyObject *obj);
void jm_patchjobid(jm_job *jp, int id);
