# Bring in configure based make vars
include ../../config/make.conf

BINDIR=../../bin
LIBDIR=../../lib
INCDIR=../../include
DEFLIST=
HOST=$(shell expr "$(HOSTNAME)" : "\([^ -]*\)")
OS=$(shell uname -s)
# Figure out what the jm_worker shared library is called
ifeq ($(OS),Darwin)
	JMLIB=libjm_worker.dylib
	JMOMPLIB=libjm_workeromp.dylib
else
	JMLIB=libjm_worker.so
	JMOMPLIB=libjm_workeromp.so
endif
JMLIBS=libjm_worker.a
JMOMPLIBS=libjm_workeromp.a

CXX=mpic++ -std=c++14
CC=mpicc -std=c99

INCLUDES=-I../shared
CXXFLAGS=$(INCLUDES) $(DEFLIST) -Wall -g -O0 -fno-omit-frame-pointer
CFLAGS=$(INCLUDES) $(DEFLIST) -Wall -g -O0 -fno-omit-frame-pointer
LDFLAGS:=-L$(MPI_LIB_DIR)

all: $(BINDIR)
	make $(BINDIR)/demoworker

$(BINDIR):
	mkdir -p $(BINDIR)
	chmod ug+rw $(BINDIR)
$(LIBDIR):
	mkdir -p $(LIBDIR)
	chmod ug+rw $(LIBDIR)

$(BINDIR)/demoworker: demoworker.cc ../shared/jm.h $(LIBDIR)/$(JMLIB)
ifeq ($(OS),Darwin)
	# In Mac OS/X the relative path notion gets embedded in the shared library when it is built.
	# use otool -L to inspect libs and executables
	# install_name_tool -change <new> <old> can be used to change to a relative location after the fact.
	# using <new>="@loader_path/.../libjm_worker.dylib"
	# <old> would be whatever was used for -install_name in shared library construction.
	$(CXX) -o demoworker $(CXXFLAGS) -rdynamic demoworker.cc -L../../lib -ljm_worker
else
	# link with libjm_worker so that it can be found relative to the executable in <execdir>/../lib
	# This will allow us to drop new libjm_worker.so files into the chroma tree.
	# https://stackoverflow.com/questions/6323603/ld-using-rpath-origin-inside-a-shared-library-recursive
	# $(CXX) -o demoworker $(CXXFLAGS) demoworker.cc -Wl,-rpath,"\$$ORIGIN/../lib,-rpath-link,." -l:libjm_worker.so
	$(CXX) -o demoworker $(CXXFLAGS) $(LDFLAGS) -rdynamic demoworker.cc -Wl,-rpath,"\$$ORIGIN/../lib,-rpath-link,." -L. -L../../lib -ljm_worker
endif
	install -m 755 demoworker $(BINDIR)

clean:
	/bin/rm -f .depend
	/bin/rm -f $(BINDIR)/demoworker *.o

.depend depend:
	$(MPICXX) -MM $(CPPFLAGS) $(CXXFLAGS) demoworker.cc > .depend

include .depend
