
CC=mpicc
CXX=mpic++

CCFLAGS=
CXXFLAGS=$(CCFLAGS)

LDFLAGS:=
ifdef MPI_DIR
ifneq ($(wildcard $(MPI_DIR)/lib.),)
LDFLAGS+=-L$(MPI_DIR)/lib
else
LDFLAGS+=-L$(MPI_DIR)/lib64
endif
endif

MPIRUN=mpirun
ifdef CRAY_MPICH_PREFIX
ismpich=1
ismvapich=0
hostargs=-f hosts.txt
MPIRUN=srun -N 1
else
mrun=$(shell which mpirun)
# crude way of figuring out which mpi we are running
ismpich=$(findstring mpich,$(mrun))
ismvapich=$(findstring mvap,$(mrun))
ifneq (,$(ismpich))
hostargs=-f hosts.txt
else ifneq (,$(ismvapich))
hostargs=-f hosts.txt
else
hostargs=-hostfile hosts.txt
endif
endif

all: worker master

tstworker: worker.c
	$(CC) -std=c99 $(CCFLAGS) $(LDFLAGS) -o worker worker.c

tstmaster: master.c
	$(CC) -std=c99 $(CCFLAGS) $(LDFLAGS) -o master master.c

rankpool: rankpool.c
	$(CC) -std=c99 $(CCFLAGS) $(LDFLAGS) -o rankpool rankpool.c

clean:
	rm -f worker master rankpool hosts.txt

.PHONY: hosts.txt
hosts.txt:
ifneq (,$(ismpich))
	hostname | sed 's/$$/:8/' > hosts.txt
else ifneq (,$(ismvapich))
	hostname | sed 's/$$/:8/' > hosts.txt
else
	hostname | sed 's/$$/ slots=8/' > hosts.txt
endif
	

# Run a simple test using MPI_Abort in the child.   
# Master should survive to exit normally
test: hosts.txt tstmaster tstworker
	$(MPIRUN) $(hostargs) -n 2 ./tstmaster arg1 arg2
# Run a simple test using a segmentation violation in the child.   
# Master should survive to exit normally
segv: hosts.txt tstmaster tstworker
	$(MPIRUN) $(hostargs) -n 2 ./tstmaster -segv arg2
