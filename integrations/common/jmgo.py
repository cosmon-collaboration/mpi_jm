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
    jm_common.startup(None)
    mpi_jm.sortjobs()
    jm_common.printJobs()

def schedInit():
    argv = sys.argv
    argc = len(argv)
    i = 1
    print("jmgo: processing args", flush=True)
    while i < argc:
        if argv[i] == '-help' or argv[i] == '--help':
            usage()
            quit()
        elif argv[i] == '-jy':
            # Option to pass startup yaml as command line option
            i = i + 1
            if i >= argc:
                raise ValueError("missing path for -jy switch")
            print("jmgo: Found startup yaml file {argv[i]}")
            os.environ['JM_LAT_STARTUP_YAML'] = argv[i]
            i = i + 1
        else{
            print(f"   args {i}: {argv[i]}", flush=True)
            i = i + 1
    main()
