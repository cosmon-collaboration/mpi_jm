#
# support eos jobs
#
#
import traceback
import logging
import sys
import time
import mpi_jm
import jm_machine
import jm_common
import pathlib

# jm_machine determined number of gpus per node
gpus_per_node=jm_machine.gpus_per_node
cores_per_node=jm_machine.cores_per_node

# GLOBAL VARIABLES
executable=None 

# base runtime table for 4th order diagrams
# All these tables are based on 4 nodes
d5th_runtime = {
    -1 : 1000,
    0 : 1000,
    1 : 1000,
    2 : 1000,
    3 : 1000,
    4 : 1000
}
d4th_runtime = {
    -1: 1000,
    0 : 243,
    1 : 4482,
    2 : 5553,
    3 : 220
}

d3rd_runtime = {
    -1: 200,
    0 : 200,
    1 : 250,
    2 : 200,
    3 : 200
}

d2nd_runtime = {
    -1: 120
}

d3b2nd_runtime = {
    -1: 1*60
}
d3b3rd_runtime = {
    -1: 15*60
}

dHf_runtime = {
    -1: 60   # kensho
}

# Estimate the runtime from the runtime tables
# above
# TODO: include target precision of result.
# kensho result is with target  0.020
# runtime goes as square of   0.020/target
def est_runtime_sub(diag, epsabs):
    # num is a group number
    ds = diag.split(':')
    if len(ds) == 2:
        ds.append(-1)
    name,ordopt,num = ds
    num = int(num)
    if name == "2Body_5th":
        return d5th_runtime[num]
    if name == "2Body_4th":
        return d4th_runtime[num]
    if name == "2Body_3rd":
        return d3rd_runtime[num]
    if name == "2Body_2nd":
        return d2nd_runtime[num]
    if name == "2Body_HF":
        return dHf_runtime[num] * (0.020 / epsabs)**2
    if name == "3Body_2nd":
        return d3b2nd_runtime[num]
    if name == "3Body_3rd":
        return d3b3rd_runtime[num]
    print("ds=", ds)
    raise ValueError('unknown diagram ' + diag)

#
# estimate and set job runtime. 
# adjust resources to control runtime
#
def est_runtime(j, diag, epsabs):
    print(f"estimate runtime {j.name} {diag}", flush=True)
    nnodes = 1
    if "2Body_5th" in diag:
        nnodes = 64
        s = 100
    elif "2Body_4th" in diag:
        nnodes = 16 
        s = 100
    elif "2Body_3rd" in diag:
        nnodes = 8
        s = 100
    elif "2Body_2nd" in diag:
        nnodes = 8
        s = 100
    elif "2Body_HF" in diag:
        nnodes = 1
        s = 100
    elif "3Body_2nd" in diag:
        nnodes = 1
        s = 2*60
    elif "3Body_3rd" in diag:
        nnodes = 2
        # 35 minutes on one node.
        s = (5 + 30 // nnodes) * 60
    else:
        nnodes = 4
        s = 1.0 * est_runtime_sub(diag, epsabs)
        if s > 400:
            nnodes = 4
            # 20% is fixed overhead
            s = 0.6 * s 
    # if running in a small block, reduce node count to match
    # and scale up runtime.
    blocksize=mpi_jm.getblocksize()
    if nnodes > blocksize:
        s *= nnodes / blocksize
        nnodes = blocksize
    # could implement system scaling here
    # need to get GPUs/node for mult
    if gpus_per_node > 0:
        ranks_per_node=gpus_per_node
        print(f"gpus_per_node = {gpus_per_node}, ranks={gpus_per_node}", flush=True)
        # If we have gpus, then one rank per gpu, 
        # and threading for the rest of the cores
        j.setresources("cpu|gpu", nnodes*gpus_per_node, 1)
    else:
        # If we just have cores, then use 16 threads/rank
        ranks_per_node=cores_per_node//16
        print(f"cores_per_node = {cores_per_node}, ranks={ranks_per_node}", flush=True)
        j.setresources("cpu", nnodes*ranks_per_node, 1)
    sec = int(round(s))
    j.estTime = sec
    j.addenv("OMP_NUM_THREADS", str(cores_per_node//ranks_per_node))
    print("Set job estTime to ", sec, " : ", j.estTime, flush=True)


#
# Returns None on sucess, msg on abort
#
def startcmd(j):
    print("Startcmd: ", j.name, flush=True)
    et = time.time() + j.estTime
    if et > jm_common.jm_end_time:
        print("time=", time.time(), ", estTime=", j.estTime, ", et=", et)
        print("jm_end_time=", jm_common.jm_end_time, flush=True)
        return "Job " + j.name + " will not complete before the allocation end time, skipping ..."
    # make sure there isn't another allocation that has started the job
    print("Startcmd: trying to claim job", flush=True)
    r = None
    # r = jm_common.claimJob(j) # try to make <jobfile>.lock file
    print("Startcmd: job claimed, r=", r,  flush=True)
    if r is None: # no problems, we claimed the job
        # make sure output dir path is created
        print("jm_eos.startcmd: Making output dir: ", j.wdir, flush=True)
        if True:
            mpi_jm.mkdirp(j.wdir) # make sure work directory exists
        else:
            # This path is crashing inside the wdir.mkdir
            # I think it only fails on ppc (summit/sierra are examples)
            wdir = pathlib.Path(j.wdir)
            wdir.mkdir(parents=True, exist_ok=True)
        print("jm_eos.startcmd: pathlib wdir.mkdir() done ", flush=True)
        # Check for final log file again now that we hold the lock
        logpath = wdir.parent / (j.name.stem + ".log")
        print("jm_eos.startcmd: logpath: ", logpath)
        if logpath.exists():
            print("jm_eos.startcmd: logfile already exists, aborting", flush=True)
            return "Job " + j.name + " has already been run"
        else:
            print("jm_eos.startcmd: startcmd successful", flush=True)
    return r

# Called at job completion (error or otherwise)
#
# TODO:  We want to move the file
# wdir / (j.name.stem + ".logtmp") to
# wdir / (j.name.stem + "log")
# 
def wrapcmd(j):
    # msg = j.rename_logtmp()
    print("Wrapping ", j.name, ", Success!", flush=True)

#
# Check if job is at minimum density,
#   if so, then we double the nStart value, finish the cmd and queue it.
# Otherwise, we look for density - 0.01 and see if there is a training output file ".to"
#   if so, then we use it to finish the cmd and queue it
# Otherwise, we see if density - 0.01 has been run without a .to file
#   if so, then we 
#
def depcmdsub(j):
    d = j.dict  # get dictionary
    diag = d['diag']
    density = d['density']      # string
    nStart = d['nStart']        # integer
    nIncrease = d['nIncrease']  # integer
    group = d['group']          # integer
    usegroups = False
    tosuf = ".to"
    tstgrp = ""
    bottomd = "0.05"
    ddelta = 0.01
    splitgroups = False
    if "2Body_5th" in diag:
        usegroups = True
        splitgroups = False
        tstgrp = ".g" + str(group) + "."  # Test for matching group
        dn = float(density)
        if dn > 0.121 and  dn < 0.201: # 0.13 .. 0.12 borrow training
            bottomd = "0.00"  # we will borrow
            ddelta = 0.01     # borrow from immediately preceeding density
        else:
            bottomd = "0.08" #  borrow from 4 down unless at starting density 0.08
            ddelta = 0.04
    elif "2Body_4th" in diag:
        nStart = nStart // 2
        nIncrease = nIncrease // 2
        usegroups = True
        tstgrp = ".g3."  # test for last group training output
        ddelta = 0.02
        bottomd = "0.06"
    elif "2Body_3rd" in diag:
        nStart = nStart // 8
        nIncrease = nIncrease // 8
        usegroups = True  # there are two groups
        tstgrp = ".g1."  # test for last group training output
    elif "2Body_HF" in diag:
        usegroups = False
    elif "3Body_3rd" in diag:
        usegroups = True
    # if 3rd order and up
    # note: %g tells eos to substitute the current group id
    if usegroups:
        tosuf = ".g%g.to"
        topathstr = j.getpathsuf(".g%g.to")  # <outputdir>/<jobstem>.g<groupid>.to
        selfto = topathstr.replace(".g%g.", tstgrp)
    else:
        topathstr = j.getpathsuf(".to")  # <outputdir>/<jobstem>.g<groupid>.to
        selfto = topathstr

    # check for self ti file.   Could have aborted after writing training file and
    # we can start with it.
    selftopath = pathlib.Path(selfto)

    foundti = False
    if selftopath.exists():
        foundti = True
        j.addarg("-ti")
        j.addarg(topathstr)  # topathstru has the %g if groups are used
    elif density == bottomd:
        # at bottom, so there won't be a training output below in density
        nStart  = nStart + (nStart // 2)
        nIncrease = nIncrease + (nIncrease // 2)
    else:
        cdensity = "_d" + density + "_"
        mdensity = "_d%0.2f_" % (float(density) - ddelta) # next lower density string
        # training in is training out for next smaller density
        tipathstr = topathstr.replace(cdensity, mdensity)
        if usegroups:
            # replace group pattern with last group string.  We assume
            # that if the last group training file is on disk then all are.
            tipathtest = tipathstr.replace(".g%g.", tstgrp)
        else:
            tipathtest = tipathstr
        path = pathlib.Path(tipathtest)
        if path.exists():
            # found suitable training output
            foundti = True
            j.addarg("-ti")
            j.addarg(tipathstr)  # for groups will have .g%g. in it to keep training files separated
        else:
            # get log file for lower density
            tilogstr = j.getpathsuf(".log")
            tilogstr = tilogstr.replace(cdensity, mdensity)
            path = pathlib.Path(tilogstr)
            if path.exists():
                # job has been run without producing training output
                # that makes this case a boundary case.
                nStart *= 2
                nIncrease *= 2
            else:
                return  # not ready to run, leave in constructing state
    if foundti:
        print("Sched: Activating job with training in ", j.name, flush=True)
    else:
        print("Sched: Activating job ", j.name, flush=True)
    j.addarg("-to")
    j.addarg(topathstr)
    j.addarg("-mc.nStart")
    j.addarg(str(nStart))
    j.addarg("-mc.nIncrease")
    j.addarg(str(nIncrease))
    j.queue() # done - move to waiting state
    return ""

def depcmd(j):
    # print(f"Sched: Depcmd on job {j.name}, state={j.state}",  flush=True)
    try:
        return depcmdsub(j)
    except Exception as e:
        logging.error(traceback.format_exc())
        return "Exception"

def init_mod(cfg):
    global executable
    print("One time init for module jm_eos - Version 2")
    val = cfg["executable"]
    executable = jm_common.fixRelPath(val)
    print("eos init mod: exe at ", executable)

def add_job(jdict):
    global executable
    print("In jm_eos.add_job!")
    print(jdict)
    ypath = jdict['yamlpath'] # pathlib Path object
    odir = jdict['odir']     # Place to run job
    rsltpath = odir / (ypath.stem + ".log")
    diag = jdict['diag']
    if rsltpath.exists():
        print(f"Result {rsltpath.as_posix()} already exists, skipping job")
        return
    j = mpi_jm.job() # create job
    d = dict()
    if 'group' in jdict:
        d['group'] = jdict['group']
    else:
        d['group'] = -1
    dens = jdict['density']
    d['density'] = dens
    j.priority = 100 - int(float(dens) * 100.0 + 0.5)
    d['nStart'] = jdict['nStart']
    d['nIncrease'] = jdict['nIncrease']
    d['diag'] = diag
    j.dict = d # save dict on job
    j.addenv("SLURM_CLUSTER_NAME", "1") # This appears to let qplib know that it is run under MPI
    j.nodeMemKb = mpi_jm.getnodemem() // 100
    j.name = ypath.as_posix()
    j.wdir = odir.as_posix()
    j.startcmd = "jm_eos.startcmd"  # last chance just as job is to be launched
    j.wrapcmd = "jm_eos.wrapcmd"    # call at end of job
    j.depcmd = "jm_eos.depcmd"      # call to see if more jobs have dependencies satisfied.
    j.addenv("PGPU_NOISY", "1")
    j.addenv("JM_EOS_SET_AFFINITY", "1")
    # j.addenv("EOS_NNEFFPWBASEPATH", "/home/chiral/NN_matrix_elements_SRG")
    # j.addenv("EOS_NNPWBASEPATH","/home/chiral/NN_matrix_elements_SRG")
    # j.addenv("EOS_PARAMSDIR", "/home/ken/EOS_DEV/eos/params")
    epsabs=jdict['epsabs']
    # set resources and estimated time
    est_runtime(j, diag, float(epsabs))
    pf = jdict['pf']
    print(f"Setting pgm={executable} for {j.name}")
    j.pgm = executable
    j.addarg("-dd")
    if gpus_per_node > 0:
        j.addarg("-gpu")
    j.addarg("-pvegas")
    j.addarg("-rot3")
    j.addarg("-h")
    j.addarg(jdict['ham'])
    # j.addarg("-t")
    # j.addarg(str(jdict['temperature']))
    j.addarg("-pf")
    j.addarg(pf)
    # j.addarg("-jmax")
    # j.addarg(str(jdict['jmax']))
    j.addarg("-epsImagSpe")
    j.addarg("1e-4")
    j.addarg("-epsabs")
    j.addarg(epsabs)
    j.addarg("-epsdiag") # regulator values
    j.addarg(jdict['epsdiag'])
    j.addarg("-no3n")
    j.addarg(jdict['no3n'])
    j.addarg("-no")
    # We don't use preno (async pre launch of normal ordering) with 
    # 3Body residual diagrams.
    if not ("3Body" in diag):
        j.addarg("-preno")
    j.addarg("-density")
    j.addarg(jdict['density'])
    j.addarg("-diag")
    j.addarg(diag)

    if True:
        # Still need to modify eos to use tmp file for training out
        # and rename at the end.
        depcmd(j)
    else:
        # Move 
        j.addarg("-mc.nStart")
        j.addarg(str(jdict['nStart']))
        j.addarg("-mc.nIncrease")
        j.addarg(str(jdict['nIncrease']))
        j.queue()
        return j
    return j
