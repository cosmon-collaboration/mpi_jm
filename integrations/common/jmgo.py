#
# Startup code for the python frontend for mpi_jm
#
import sys
import time
import mpi_jm
import jm_common
import pathlib
import yaml

def main():
    jm_common.startup(None)  # Scans for and loads job files
    mpi_jm.sortjobs()        # priority sort.
    jm_common.printJobs()    # log them to scheduler log file

def usage():
    print("Usage: python3 -m startjm -lumpsize <allocationsize> -blocksize <blocksize> -py jmgo <startup>.yaml")
    print("jmgo.py is the scheduler startup script for application integrations")
    print("  -help   - This message")
    print("  <startup>.yaml  where <startup> is the configuration file for the batch submission")

def schedInit():
    '''
    schedInit will be called by jm_sched after startup.
    We set up the config file, call the standard startup (jm_common.startup) and
    then sort and print the jobs to the scheduler log file.
    On return, the scheduler starts work.
    '''
    argv = sys.argv
    argc = len(argv)
    i = 1
    print("jmgo: processing args", flush=True)
    startyaml = "" # false in if
    while i < argc:
        if argv[i] == '-help' or argv[i] == '--help':
            usage()
            quit()
        elif argv[i].endswith(".yaml"):
            if startyaml:
                raise ValueError(f"Found second startup yaml file {startyaml}, then {argv[i]}")
            startyaml = argv[i]
            # Must be the startup file
            # Option to pass startup yaml as command line option
            print("jmgo: Found startup yaml file {startyaml}", flush=True)
            # stick in environ to be picked up later
            os.environ['JM_LAT_STARTUP_YAML'] = startyaml
            i = i + 1
        else:
            print(f"   skipping args {i}: {argv[i]}", flush=True)
            i = i + 1
    main()
