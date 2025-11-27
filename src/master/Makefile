# These are where we store files that will have to be installed later
BINDIR=../../bin
LIBDIR=../../lib
INCDIR=../../include

DEFLIST=
HOST=$(shell expr "$(HOSTNAME)" : "\([^ -]*\)")
OS=$(shell uname -s)

CXX=mpic++ -std=c++14 -DMVAPICH2=1
CC=mpicc -std=c99

INCLUDES=-I../shared
CXXFLAGS=$(INCLUDES) $(DEFLIST) -Wall -g -O0 -fno-omit-frame-pointer
CFLAGS=$(INCLUDES) $(DEFLIST) -Wall -g -O0 -fno-omit-frame-pointer
ifdef DISCONNECT_CHILD
	CXXFLAGS+=-DJM_DISCONNECT_CHILD
	CFLAGS+=-DJM_DISCONNECT_CHILD
endif
LDFLAGS:=
ifdef MPI_DIR
ifneq ($(wildcard $(MPI_DIR)/lib.),)
LDFLAGS+=-L$(MPI_DIR)/lib
else
LDFLAGS+=-L$(MPI_DIR)/lib64
endif
endif

all: $(BINDIR)
	make -e $(BINDIR)/jm_master

$(BINDIR):
	mkdir -p $(BINDIR)
	chmod ug+rw $(BINDIR)

$(BINDIR)/jm_master: jm_master.cc ../shared/jm.h ../shared/jm_int.h
	$(CXX) -g -O0 -rdynamic  $(LDFLAGS) -o $(BINDIR)/jm_master $(CXXFLAGS) jm_master.cc

clean:
	/bin/rm -f $(BINDIR)/jm_master *.o
	/bin/rm -rf worker worker.dSYM jm_master jm_master.dSYM
