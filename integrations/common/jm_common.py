# BSD 3-Clause License
# Copyright (c) 2025, The Regents of the University of California
# See Repository LICENSE file
import os
import sys
import time
import mpi_jm
import importlib
import yaml
import pathlib
import logging

# File Locking
# See second answer to: 
#  https://stackoverflow.com/questions/10978869/safely-create-a-file-if-and-only-if-it-does-not-exist-with-python/18474773#18474773
#  Basically one uses   f = open('lockfile', 'x')
#  which opens exclusively in write mode.
# Issue:  if directory is not writeable then lockfile can't be opened either
#         Figure out how to distinguish
#  

cfg = None  # Global config file
reldir = None
logger = logging.getLogger(__name__)
jm_start_time = time.time()  # floating point seconds
jm_end_time = jm_start_time + 100000000000000
setenv_list = []

# dictionary of modules for handling job creation
moddict = dict()

def getmachine():
    h = os.environ.get('LMOD_SYSTEM_NAME', '') # for summit
    if h != '':
        return h
    h = os.environ.get('LSB_EXEC_CLUSTER', '') # for sierra
    if h != '':
        return h
    h = os.environ.get('LCSCHEDCLUSTER', '') # for lassen
    if h != '':
        return h
    # Fall back to OS name
    return os.uname()[0]

def fixRelPath(s):
    # if s has {ENVNAME} in it, this will apply the dictionary
    s = s.format(**os.environ)
    # special case {JM_RELPATH} without putting in env
    s = s.format({'JM_RELPATH': reldir})
    # Expand things like ~/, ~user
    s = pathlib.Path(s).expanduser().as_posix()
    return s

machine = getmachine()

#
# Mark job as owned by this run
# to be called from startcmd callback
#
def claimJob(j):
    lockpath = pathlib.Path(j.jobfile).with_suffix(".lock")
    if lockpath.exists():
        return "Claimed"
    r = None
    try:
        lockpath.touch(exist_ok=False)
    except FileExistsError:
        r = "Claimed"
    except:
        r = "Can't create lock file " + lockpath.as_posix()
    return r

def addJobFailed(jdict, msg):
    global logger
    print("Job creation failed: ", msg, flush=True)
    logger.error("Job creation failed: " + msg)
    logger.info(jdict)

# use 'type' entry to select module
def addJob(jdict):
    global logger
    yamlpath = jdict['yamlpath']
    if not 'type' in jdict:
        addJobFailed(jdict, "Ignoring job dictionary missing 'type' field")
        return
    ts = jdict['type']
    modstr = 'jm_' + ts
    if modstr in moddict:
        mod = moddict[modstr]
    else:
        try:
            mod = importlib.import_module(modstr)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"No installed module {modstr} to handle job {yamlpath}")
        moddict[modstr] = mod
        print(f"modstr = {modstr}", flush=True)
        print(f"cfg = {cfg}", flush=True)
        modcfg = {}
        if modstr in cfg:
            modcfg = cfg[modstr]
        print(f"module cfg = {modcfg}", flush=True)
        mod.init_mod(modcfg)
    try:
        mod.add_job(jdict)
    except Exception as e:
        # Todo: Add cleanup of unfinished jobs
        print(f"Failure adding job {yamlpath}")
        print(f"Exception {e}")
        logger.exception("addJob:")
        addJobFailed(jdict, "Unknown error while adding job {yamlpath}")

# Using the dictionary describing the job, construct the
# output directory.
def getOutDir(d):
    f = d['yamlpath'] # pathlib object
    dir=f.parent      # containing directory
    stem=f.stem       # string - the local yaml file minus the ".yaml"
    stem += ".log"    # eventual log file name

def addJobFile(f, od):
    try:
        stream = open(f, 'r')
    except:
        print("can't open ", f, flush=True)
        return
    try:
        for d in yaml.load_all(stream, Loader=yaml.FullLoader):
            print("dict = ", d, flush=True)
            d['yamlpath'] = f # save pathlib object
            d['odir'] = od    # location for result files.
            j = addJob(d)
            if(j != None):
                j.jobfile = f # make sure we record where job came from
    except yaml.scanner.ScannerError as e:
        print(f"Yaml scanning error loading {f}, msg={e}")
    except Exception as e:
        print(f"Unknown error loading {f}, msg={e}")
    return

def printJob(j):
    ts="     "
    print("job ", j.name, ":")
    print(ts, " state=", j.state)
    print(ts, " wdir=", j.wdir)
    print(ts, " memKb=", j.nodeMemKb)
    print(ts, " estTime=", j.estTime)
    print(ts, " pgm=", j.pgm)
    print(ts, " args=", j.getargs())
    env = j.getenv()
    print(ts, " env=", env)
    print(ts, " startTime=", j.startTime)
    print(ts, " endTime=", j.endTime)
    print(ts, " resources=", j.getresources())
    print(ts, " startcmd=", j.startcmd)
    print(ts, " wrapcmd=", j.wrapcmd)
    print(ts, " depcmd=", j.depcmd)
    print(ts, " priority=", j.priority)
    print(ts, " jobfile=", j.jobfile)
    print(ts, " dict=", j.dict, flush=True)
    # sys.stdout.flush()

def printJobs():
    for j in mpi_jm.jobs():
        printJob(j)

# Load yaml files from directory d
def loadJobDir(cfg, d, od):
    print(f"Loading jobs from {d}")
    print(f"Result files placed in {od}", flush=True)
    members = pathlib.Path(d)
    for f in members.iterdir():
        if f.is_dir():
            if cfg['recursive']:
                loadJobDir(cfg, f, od / f.name)
        elif f.match('*.yaml'):
            if not f.name.startswith('.'): # skip hidden files
                print(f"entry: {f.name}", flush=True)
                addJobFile(f, od)
        else:
            print(f"unknown file: {f.name}", flush=True)


#
# Figure out when the allocation ends so we can test
# jobs to see if they should launch.
#
def set_end_time():
    global jm_end_time, jm_start_time, jm_alloc_time
    argv = sys.argv
    argc = len(argv)
    jm_alloc_time = 100000000000.0
    i = 1
    while i < argc:
        if argv[i] == "-alloctime":
            i = i + 1
            jm_alloc_time = float(argv[i])
        i = i + 1
    jm_end_time = jm_start_time + jm_alloc_time
    print("jm_alloc_time = ", jm_alloc_time)
    print("jm_start_time = ", jm_start_time)
    print("jm_end_time   = ", jm_end_time, flush=True)

#
# We've been given a list of expressions to expand
# into env vars for launched processes.
# We are setting them in the environment of jm_sched,
# not jm_master, so they won't actually propagate unless
# we somehow copy them there.
#
def process_setenv():
    print("Processing setenv list from setup yaml file", flush=True)
    global cfg, setenv_list
    envlist = cfg.get("setenv", [])
    for e in envlist:
        if not isinstance(e,str):
            print(f"Ignoring non string setenv: {e}", flush=True)
        else:
            i = 0
            for i in range(len(e)):
                if e[i] == '=':
                    name = e[0:i].strip()
                    val = e[i+1:].strip()
                    val = val.format(**os.environ)
                    os.environ[name] = val
                    print(f"   Setting env {name} to {os.environ[name]}", flush=True)
                    setenv_list.append([name, val])
                    break
            if i >= len(e):
                print(f"   Ignoring incorrect format setenv str: {e}", flush=True)

def startup(jpath):
    global logger
    global cfg, reldir
    if(jpath == None):
        jpath = os.getenv('JM_LAT_STARTUP_YAML')
    if(jpath == None):
        print("No startup yaml file specified by env JM_LAT_STARTUP_YAML, or argument", flush=True)
        return
    print("Running on: ", machine, flush=True)
    set_end_time()
    # attempl load of configuration yaml file
    reldir = pathlib.PurePath(jpath).parent
    stream = open(jpath, 'r')
    cfg = yaml.safe_load(stream)
    stream.close()
    process_setenv()
    if not 'dirs' in cfg:
        print("Startup yaml '", jpath, "' is missing dirs list", flush=True)
        raise KeyError("dirs")
    # defaults for configuration
    if not 'recursive' in cfg:
        cfg['recursive'] = False
    logfilename = cfg.get('logfile', 'jm_lat.log')
    # logger = logging.basicConfig(filename=logfilename)

    # Now search the dirs
    dirs = cfg['dirs']
    dlen = len(dirs)
    if 'odirs' in cfg:
        odirs = cfg['odirs']
        if len(odirs) != dlen:
            raise ValueError("config 'odirs' does not match dirs in length")
    else:
        # write output in same dir as input yaml files
        # May want to disable this
        odirs = dirs
    for di in range(dlen):
        dd = pathlib.Path(fixRelPath(dirs[di]))
        od = pathlib.Path(fixRelPath(odirs[di]))
        loadJobDir(cfg, dd, od)
