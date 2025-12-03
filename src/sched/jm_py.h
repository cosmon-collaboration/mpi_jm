/*
 BSD 3-Clause License
 Copyright (c) 2025, The Regents of the University of California
 See Repository LICENSE file
 */
#include <Python.h>
#include "structmember.h"
void jm_py_init();
char *get_py_str(PyObject *value);

int jm_pyjob2jobx(PyObject *obj);
jm_job *jm_pyjob2job(PyObject *obj);
void jm_patchjobid(jm_job *jp, int id);
