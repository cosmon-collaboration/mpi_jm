# Spawn Test

The point of this test is to check that a disconnected child (via spawn) can abort and not take down the parent.
We need this for mpi_jm becauses the managed jobs can abort for many reasons.

worker.c generates tstworker which is the client that is spawn’d and will call MPI_Abort or use a segmentation violation to abort the child.    

master.c generates tstmaster which is the parent that will do the spawn.

The Makefile has two tags: test and segv to perform testing.

This test also uses a shell script as part of the launch to pass environment variables to the child.   Some MPI implementations have a ridiculously	small buffer to store the set of environment variables so script jm_spawnwrap is used to pass them through the argument list instead.

# Building

```
 # builds tstworker and tstmaster
 % make 
```

The launch script jm_spawnwrap is in the same directory.

# Running tests

## Abort

The first test verifies that an abort of a disconnected child will not abort the parent.

```  
 # launches mpirun -n 2 ./tstmaster arg1 arg2  
 % make test 
```  

The arguments are to verify that tstworker receives arguments through the spawn.

The master waits ten seconds and writes a success message.  This gives the worker time to start and do the abort.  

## Segv

The second test verifies that a segmentation violation (or other signal) in the child will not abort the parent.

```
 # launches mpirun ./tstmaster -segv arg2
 % make segv 
```

The master waits ten seconds and writes a success message.  This gives the worker time to start and force a segmentation fault.  

# Results

## Mvapich-plus-4.1

Abort is handled correctly.    The error does not propagate.

A segmentation fault incorrectly propagates to the master and takes the job down.

We can work around this until a fix is supplied by having the client library insert a segv and bus error handler that does an Abort.

