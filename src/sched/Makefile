# Include configure based make vars
include ../../config/make.conf

BINDIR=../../bin
LIBDIR=../../lib
PYLIBDIR=../../pylib
HOST=$(shell expr "$(HOSTNAME)" : "\([^ -]*\)")
DEFLIST=

# Find python library
PYINC=$(shell python3-config --includes)
PYLIB=$(shell if (python3-config --embed > /dev/null 2>&1); then python3-config --embed --ldflags; else python3-config --ldflags; fi | sed -e "s/-framework.*//")

INCLUDES=-I../shared $(PYINC)
CPPFLAGS=
CXXFLAGS=$(INCLUDES) -Wall -g -O0 -fno-omit-frame-pointer
# CXXFLAGS=$(INCLUDES) -Wall -g -O3 -Wall

CC=$(MPICC)
CXX=$(MPICXX)

LDFLAGS:=-L$(MPI_LIB_DIR) $(CFGLINKFLAGS)

# SRCS=jm_py.cc jm_sched.cc jm_res.cc
SRCS=jm_py.cc jm_sched.cc jm_res.cc jm_lump.cc jm_log.cc jm_mkdirp.c
OBJS=jm_py.o jm_res.o jm_lump.o jm_log.o jm_mkdirp.o
TOBJ=jm_sched.o # don't include in list

# all: $(BINDIR) $(LIBDIR) $(PYLIBDIR)/jm_machine.py $(PYLIBDIR)/jobs.py $(PYLIBDIR)/cjobs.py
all: $(BINDIR) $(LIBDIR)
	make -e $(BINDIR)/jm_sched 

$(BINDIR):
	mkdir -p $(BINDIR)
	chmod g+rwx $(BINDIR)
$(LIBDIR):
	mkdir -p $(LIBDIR)
	chmod g+rwx $(LIBDIR)

jm_sched.o: jm_sched.cc
jm_py.o: jm_py.cc
jm_res.o: jm_res.cc

$(BINDIR)/jm_sched:  $(OBJS) jm_sched.o
	$(MPICXX) -g $(LDFLAGS) -O0 -o $(BINDIR)/jm_sched $(CXXFLAGS) jm_sched.cc $(OBJS) $(PYLIB)

clean:
	/bin/rm -f $(BINDIR)/jm_sched *.o

.depend depend:
	$(MPICXX) -MM $(CPPFLAGS) $(CXXFLAGS) $(SRCS) > .depend

include .depend
