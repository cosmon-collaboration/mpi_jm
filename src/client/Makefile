# Bring in config variables
include ../../config/make.conf

# These are staging areas before installing
BINDIR=../../bin
LIBDIR=../../lib
INCDIR=../../include

DEFLIST=
# Figure out what the jm_worker shared library is called
ifeq ($(OS),Darwin)
	JMLIB=libjm_worker.dylib
	JMOMPLIB=libjm_workeromp.dylib
	ifeq ($(MACHINE),arm64)
		CPUID:=
	else
		CPUID:=-DUSE_CPUID=1
	endif
# Figure out what the jm_worker shared library is called
else
	CPUID:=-DUSE_SCHED=1
	JMLIB=libjm_worker.so
	JMOMPLIB=libjm_workeromp.so
endif
JMLIBS=libjm_worker.a
JMOMPLIBS=libjm_workeromp.a

INCLUDES=-I../shared
CXXFLAGS=$(INCLUDES) $(DEFLIST) -Wall -g -O0 -fno-omit-frame-pointer $(CPUID)
CFLAGS=$(INCLUDES) $(DEFLIST) -Wall -g -O0 -fno-omit-frame-pointer $(CPUID)
LDFLAGS:=-L$(MPI_LIB_DIR) ${CFGLINKFLAGS}

all: $(BINDIR)  $(LIBDIR)  $(INCDIR)/jm.h
	make -e $(LIBDIR)/$(JMLIB) $(LIBDIR)/$(JMOMPLIB) $(LIBDIR)/$(JMLIBS) $(LIBDIR)/$(JMOMPLIBS)

$(BINDIR):
	mkdir -p $(BINDIR)
	chmod ug+rw $(BINDIR)
$(LIBDIR):
	mkdir -p $(LIBDIR)
	chmod ug+rw $(LIBDIR)
$(INCDIR)/jm.h: ../shared/jm.h
	mkdir -p $(INCDIR)
	install -m 666 ../shared/jm.h $(INCDIR)

$(LIBDIR)/$(JMLIB): jm_worker.c ../shared/jm.h
	# -fpic for position independent code.
	#  shared library construction different on mac os/x
ifeq ($(OS),Darwin)
	$(MPICC) $(CFLAGS) -fpic -dynamiclib -install_name "@loader_path/../lib/$(JMLIB)" -current_version 1.0 -compatibility_version 1.0 $(LDFLAGS) -o $(JMLIB) jm_worker.c
else
	$(MPICC) $(CFLAGS) -fpic -shared -Wl,-soname,$(JMLIB) $(LDFLAGS) -o $(JMLIB) jm_worker.c
endif
	install -m 664 $(JMLIB) $(LIBDIR)

$(LIBDIR)/$(JMOMPLIB): jm_worker.c ../shared/jm.h
	# -fpic for position independent code.
	#  shared library construction different on mac os/x
ifeq ($(OS),Darwin)
	$(MPICC) $(CFLAGS) -fopenmp -fpic -dynamiclib -install_name "@loader_path/../lib/$(JMOMPLIB)" -current_version 1.0 -compatibility_version 1.0  $(LDFLAGS) -o $(JMOMPLIB) -DUSEOMP jm_worker.c
else
	$(MPICC) $(CFLAGS) -fopenmp -fpic -shared -Wl,-soname,$(JMOMPLIB)  $(LDFLAGS) -o $(JMOMPLIB) -DUSEOMP jm_worker.c
endif
	install -m 664 $(JMOMPLIB) $(LIBDIR)
	
$(LIBDIR)/$(JMLIBS): jm_worker.c ../shared/jm.h
	$(MPICC) $(CFLAGS) -c  $(LDFLAGS) -o jm_worker.o jm_worker.c
	ar cr $(JMLIBS) jm_worker.o
	install -m 664 $(JMLIBS) $(LIBDIR)
	/bin/rm jm_worker.o

$(LIBDIR)/$(JMOMPLIBS): jm_worker.c ../shared/jm.h
	$(MPICC) -fopenmp -DUSEOMP $(CFLAGS) -c  $(LDFLAGS) -o jm_worker.o jm_worker.c
	ar cr $(JMOMPLIBS) jm_worker.o
	install -m 664 $(JMOMPLIBS) $(LIBDIR)
	/bin/rm jm_worker.o

test:
	echo rpath="\$$ORIGIN"

clean:
	/bin/rm -f .depend
	/bin/rm -f $(JMLIB) $(JMOMPLIB) $(JMLIBS) $(JMOMPLIBS) *.o 
	/bin/rm -rf *.dSYM
	/bin/rm -f $(LIBDIR)/$(JMLIB) $(LIBDIR)/$(JMOMPLIB) $(LIBDIR)/$(JMLIBS) $(LIBDIR)/$(JMOMPLIBS)

.depend depend:
	$(MPICC) -MM $(CFLAGS) $(SRCS) jm_worker.c > .depend

include .depend
