#!/usr/tce/packages/python/python-3.6.4/bin/python3
# vim: tabstop=4 expandtab
# This file automates starting up Mpi_Jm in fault tolerant mode.
#
# Notes:
#   If using mpich:
#     https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager
#     hydra_nameserver &  # On service node <shost>
#     mpiexec.hydra ... -nameserver <shost>  <cmd> <args>
#
import sys
import os
import errno
import subprocess
import time
import shutil
import datetime

# experiments
# os.environ['MV2_USE_CUDA'] = "1"
# os.environ['MV2_USE_SHARED_MEM'] = "0"
# os.environ['MV2_USE_BLOCKING'] = "1"
# os.environ['MV2_USE_RDMA_CM'] = "1"
# Also needs MV2_RDMA_CM_CONF_FILE_PATH set to the path to a file with the local IP address to be used used.
# I'm not sure if this is supposed to be different on each node or not.

# collect the service node name (hostname)
# servicenode = socket.gethostname()   # hostname that mpirun's are done from.  Used for hydra_nameserver.
servicenode = os.uname()[1]  # hack way to get current host name
launchdelay=0 # delay between launches of lumps
enddelay=30   # delay from last lump launch to telling scheduler to take the lumps that have reported in.
polldelay=2  # time between checking mpirun status
nameserverlogfile="nameserver.log"
nameserveridfile=None # place to write connection id of ompi name server at startup

preload=None

useHydra=True

mpirun = []   # will be set in pickmpi to start of command line
mpirun_n = "-NUMRANKS"  # will be set in pickmpi to correct switch
ismvapich2 = True
mpirun_ns = [] # options to set up name server
mpirun_hf = "-f"

jm_master_path=None

# mpirun=["mpiexec.hydra", "-env", "MV2_SUPPORT_DPM", "1", "-env", "MV2_RNDV_PROTOCOL", "RGET", "-launcher", "ssh", "-nameserver", servicenode]

def usage():
    print("Usage: python3 start.py -hostfile <hostfile> -lumpsize <lumpsize> -blocksize <bnum> -py <jobfile> ...")
    print("  -hostfile <hostfile> : <hostfile> should be a simple hostfile with one host per line")
    print("                       : optional on perlmutter where it can build it's own hostfile")
    print("  -badhosts <file>     : file containing simple list of 'bad' nodes to avoid.")
    print("                       : Useful for brand new supercomputers that have flaky nodes.")
    print("  -lumpsize <num>      : <num> should be a multiple of the block size in jm_machine.py")
    print("                       : and it should also divide the number of hosts in <hostfile>.")
    print("                       : required argument.")
    print("  -blocksize <num>     : number of nodes in a block, should divide lumpsize, and");
    print("                       : be larger than the largest job requirement.  Power of 2 recommended.")
    print("  -alloctime <secs>    : Number of seconds that allocation will be allowed to run.")
    print("  -py <jobfile> <args> : <jobfile> and <args> will be passed to scheduler python interpreter.")
    print("", flush=True)
    exit(0)

# Using the path to mpiexec.hydra, locate libmpi.so
def preloadlibmpi():
    global preload
    # return  # preload is causing crash
    exepath = shutil.which("mpiexec.hydra")
    binpath = os.path.dirname(exepath)
    mpipath = os.path.dirname(binpath)
    if useHydra:
        libmpi = mpipath + "/lib64/libmpi.so"
    else:
        libmpi = mpipath + "/lib64/libmpi.so"
    # put in environment
    os.environ["LD_PRELOAD"] = libmpi
    # also set up to pass directly to mpiexec
    preload=libmpi
    print("Set LD_PRELOAD=", libmpi, flush=True)

def pickmpi():
    global mpirun, mpirun_hf, mpirun_n, mpirun_ns, ismvapich2, nameserver, nameserveridfile
    global jm_master_path
    jm_master_path = shutil.which("jm_master")
    if jm_master_path is None:
        raise RuntimeError("Can't find jm_master, make sure it is in the path")
    exepath = shutil.which("jm_sched")
    if exepath is None:
        raise RuntimeError("Can't find jm_sched, make sure it is in the path")
    else:
        print(f"Found jm_sched at {exepath}", flush=True)
    ppath = os.path.dirname(os.path.dirname(exepath)) + "/pylib"
    print("PYTHONPATH set to: ", os.environ['PYTHONPATH'], flush=True)
    # os.environ['PYTHONPATH'] = ppath
    if useHydra:
        exepath = shutil.which("mpiexec.hydra")
    else:
        exepath = shutil.which("mpirun_rsh")
    if exepath is None:   # must be Openmpi, try simple mpirun
        exepath = shutil.which("mpirun")
        if exepath is None:
            print("Can't detect type of mpi: openmpi or mvapich2", flush=True)
            exit(1)
        ismvapich2 = False
        # force setup to use rsh
        # ompirsh = ["-mca", "plm", "rsh", "-mca", "plm_rsh_agent", "ssh", "-mca", "plm_rsh_no_tree_spawn", "true", "-show-progress"]
        # ompirsh = ["-mca", "plm_rsh_agent", "ssh", "-mca", "plm_rsh_no_tree_spawn", "true", "-show-progress"]

        # Some hints from:  https://www.open-mpi.org/faq/?category=large-clusters
        # --fwd-mpirun-port enables a tree based overlay network.   Good for sparse comms
        # pmux_base_async_modex picks up endpoint info only on first message instead of everywhere.  Again
        # sparse connectivity will win.
        # ompirsh = ["--fwd-mpirun-port", "-mca", "plm_rsh_agent", "ssh", "-mca", "pmix_base_async_modex", "true", "-show-progress"]
        ompirsh = ["-mca", "plm_rsh_agent", "ssh", "-show-progress"]
        ompirsh += ["-mca", "fwd_mpirun_port", "true"] #  Incorrect usage
        ompirsh += ["-mca", "opal_set_max_sys_limits", "1"] # should it be -mca or --mca
        mpirun = [exepath]
        bdir=os.path.dirname(exepath)
        mdir=os.path.dirname(bdir)
        mpirun += ["--prefix", mdir]  # helps with sub-launches via spawn, passes location of mpi install
        mpirun += ["--oversubscribe"] # spawn fails without this.
        mpirun += ompirsh
        print("mpirun=", mpirun, flush=True)
        mpirun_ns = [] # options to set up name server
        mpirun_hf = "-hostfile"
        mpirun_n = "-np"
        # --no-daemonize, or it forks and the parent exits so we can keep
        # a handle on it.
        # -r nameserveridfile produces the info to connect that we need for mpirun calls.
        # launchnameserver will set mpirun_ns
        #
        nameserveridfile = "nameserver.id"
        # nameserver = ["ompi-server", "--no-daemonize", "-r", nameserveridfile] # Still need some arguments
        nameserver = None # in 5.0.x series ompi-server is gone.  Could use prte 
        # export OMPI_MCA_btl_openib_warn_default_gid_prefix=0
        os.environ['OMPI_MCA_btl_openib_warn_default_gid_prefix'] = "0"  # startup message disable.
        # export OMPI_MCA_orte_base_help_aggregate=0
        # export OMPI_MCA_routed=direct
        # os.environ['OMPI_MCA_routed'] = "direct"
        # export OMPI_MCA_pmix_base_async_modex=0
        # export OMPI_MCA_btl_tcp_latency=20000000
    else:
        ismvapich2 = True
        if useHydra:
            # mpirun=[exepath, "-launcher", "ssh", "--bind-to", "none", "-envall"]
            mpirun=[exepath, "-launcher", "ssh", "-envall"]
            if getmachine() == "sierraxx":
                mpirun += ["-iface", "hsi0"] # use Infiniband interface
            mpirun_hf = "-f"
            mpirun_ns = ["-nameserver", servicenode] # options to set up name server
            nameserver = ["hydra_nameserver"]
        else:
            mpirun=[exepath, "-launcher", "ssh", "-export-all"]
            # mpirun=[exepath, "-launcher", "ssh", "-export"]
            mpirun_hf = "-hostfile"
            # Need to figure out name server options for mpirun_rsh
            mpirun_ns = ["-nameserver", servicenode] # options to set up name server
            nameserver = ["hydra_nameserver"]
        print(f"mpirun = {mpirun}", flush=True)
        mpirun_n = "-np"
        # note: environment setting values are always strings
        # os.environ['MV2_SUPPORT_DPM'] = "1"  # required with mvapich2 to enable spawn/accept/connect ...
        os.environ['MV2_DEBUG_SHOW_BACKTRACE'] = "1"
        # os.environ['MV2_RNDV_PROTOCOL'] = "RGET"
        # os.environ['MV2_USE_MCAST'] = "0"
        # os.environ['MV2_USE_ALIGNED_ALLOC'] = '1'
        # os.environ['MV2_USE_RDMA_CM'] = "0"  # suggestion from Hari
        # os.environ['MV2_IBA_HCA'] = "mlx5_1:mlx5_3"
        # or
        # os.environ['MV2_IBA_HCA'] = "mlx5_1:mlx5_3"
        # os.environ['MV2_SHOW_HCA_BINDING'] = "1"
        # os.environ['MV2_NUM_HCAS'] = "2"
        preloadlibmpi()  # load from MVAPICH2 libmpi.so first
    print("mpirun=", mpirun, flush=True)


# parse arguments
def parseargs():
    argv = sys.argv
    argc = len(argv)
    if argc == 1:
        usage()
    print("args: ", argv, flush=True)
    hostfile=''
    pyargs = []
    lumpsize=0
    blocksize=0
    alloctime=0
    badhosts=""
    jm_master_v = ""
    i = 1
    while i < argc:
        if argv[i] == "-help":
            usage()
        if argv[i] == "-v":
            jm_master_v = "-v"
        elif argv[i] == "-badhosts" or argv[i] == "-bh":
            i=i+1
            badhosts = argv[i]
        elif argv[i] == "-hostfile" or argv[i] == "-f":
            i=i+1
            hostfile=argv[i]
        elif argv[i] == "-blocksize":   # number of nodes to start as a chunk
            i=i+1
            blocksize=int(argv[i])
        elif argv[i] == "-alloctime":   # number of nodes to start as a chunk
            i=i+1
            alloctime=int(argv[i])
        elif argv[i] == "-lumpsize":   # number of nodes to start as a chunk
            i=i+1
            lumpsize=int(argv[i])
        elif argv[i] == "-py":
            i=i+1
            pyargs=argv[i:]
            i = argc  # eat up all args
        else:
            print("Unknown argument: ", argv[i], flush=True)
            exit(1)
        i=i+1
    print("lumpsize=", lumpsize)
    if lumpsize==0:
        print("lumpsize not set", flush=True)
        exit(1)
    print("pyargs=", pyargs, flush=True)
    if not pyargs:
        print("Need -py jobfile ...", flush=True)
        exit(1)
    return [hostfile, badhosts, lumpsize, blocksize, alloctime, jm_master_v, pyargs]

def readhostsPerlmutter():
    '''
    Perlmutter uses Slurm, so we can use scontrol to get the hosts lists, one
    per line.
    '''
    # returns byte string
    bhoststr = subprocess.check_output(['scontrol', 'show', 'hostnames'])
    # convert to normal string and split lines at newlines
    hostlist = bhoststr.decode('utf-8').splitlines()
    # check that our list of hosts is clean
    for n in hostlist:
        if n[0:3] != 'nid':
            raise ValueError(f"Found invalid hostname {n}")
    return hostlist

def readhostsSierra():
    jobinfo=os.environ['LSB_JOBFILENAME']
    hostfile=jobinfo + ".hostfile"
    with open(hostfile) as f:
        data = f.readlines()
    data=[x.strip() for x in data]
    snode = data.pop(0)
    hostlist=[]
    last="not a host name"
    for x in data:
        if x != last:
            last=x
            hostlist.append(x)
    return hostlist

def getmachine():
    cname = os.environ.get('SLURM_CLUSTER_NAME', 'none')
    if cname == "perlmutter":
        return cname
    return os.environ.get('LSB_EXEC_CLUSTER', os.environ.get('LCSCHEDCLUSTER', ''))

def makehostpair(hostname):
    return [int("0" + "".join(c for c in hostname if c.isdigit())), hostname]
    list(map(makehostpair, hlist))

def sort_hostlist(hostlist):
    srt_lst = list(map(makehostpair, hostlist))
    srt_lst.sort()
    hosts_sorted = [h[1] for h in srt_lst]
    return hosts_sorted

# Read hostfile entries into list
def readhostssub(hostfile):
    m = getmachine()
    if hostfile:
        with open(hostfile) as f:
            hostlist=f.readlines()
        hostlist=[x.strip() for x in hostlist]
        return hostlist
    if m == "sierra" or m == "ray":
        return readhostsSierra()
    if m == "perlmutter":
        return readhostsPerlmutter()
    if os.environ['NERSC_HOST'] == "perlmutter":
        print("Running startjm from perlmutter login node, must be in debug or batch allocation", flush=True)
    else:
        print("No -hostfile specification", flush=True)
    exit(1)

def readhosts(hostfile, badhosts):
    hlist=readhostssub(hostfile)
    if badhosts:
        print("Reading bad nodes from", badhosts, flush=True)
        with open(badhosts) as f:
            blist=f.readlines()
        blist=[x.strip() for x in blist]
        for b in blist:
            print("Removing ", b, " from host list", flush=True)
            hlist.remove(b)
    return sort_hostlist(hlist)

# Compute name of host file for lump i
def lumpname(i):
    return "lump" + str(i) + ".txt"
# Log file name
def logfilename(i):
    return "lump" + str(i) + ".log"

# Write lump#.txt file with host names in lump
def writelump(i,lump):
    lf=lumpname(i)
    print(i, ": ", lf)
    with open(lf, "w") as f:
        for i in range(len(lump)):
            f.write(lump[i])
            if not ismvapich2:
                cpucnt = os.cpu_count()
                corecnt = cpucnt // 2
                f.write(f" slots={corecnt} max_slots={cpucnt}")
            f.write("\n")

# Launch a set of nodes (a lump) that will connect to the scheduler
def launchlump(i, numlumps, lumpsize, blocksize, alloctime, jm_master_v, pyargs):
    lf=lumpname(i)
    print("mpirun is ", mpirun, flush=True)
    cmd = mpirun[:] # slice for force copy
    if numlumps > 1:
        print("Adding name server", flush=True)
        cmd += mpirun_ns
    cmd += [mpirun_n, str(lumpsize), mpirun_hf, lf]
    if ismvapich2:
        # This is really for sierra
        pypath = os.environ['PYTHONPATH']
        pyubase = os.environ['PYTHONUSERBASE']
        if useHydra:
            # cmd += ["-genv", "MV2_USE_MCAST=0"]
            cmd += ["-genv", "MV2_USE_RDMA_CM=0"]
            #cmd += ["-genv", "MV2_IBA_HCA=mlx5_0"]
            #cmd += ["-genv", "MV2_NUM_HCAS=1"]
            cmd += ["-genv", "MV2_DEFAULT_TIME_OUT=30"] # big enough
            cmd += ["-genv", "MV2_SUPPORT_DPM=1"]
            cmd += ["-genv", "MV2_USE_GDRCOPY=0"]
            cmd += ["-genv", "MV2_USE_CUDA=1"]
            cmd += ["-genv", "MV2_USE_ALIGNED_ALLOC=1"]
            cmd += ["-genv", "MV2_HOMOGENEOUS_CLUSTER=1"]
            cmd += ["-genv", "MV2_SUPPRESS_NCCL_USAGE_WARNING=1"]
            cmd += ["-genv", f'PYTHONPATH={pypath}']
            if pyubase:
                cmd += ["-genv", f'PYTHONUSERBASE={pyubase}']
            cmd += ["-genv", f"LD_PRELOAD={preload}"]
            # cmd += ["-env", "MV2_IBA_HCA", "mlx5_1:mlx5_3"]
        else:
            #    cmd += ["MV2_IBA_HCA=mlx5_1:mlx5_3"]
            cmd += [
            "MV2_SUPPORT_DPM=1", "MV2_USE_CUDA=1", 
            "MV2_USE_ALIGNED_ALLOC=1", "MV2_SUPPRESS_NCCL_USAGE_WARNING=1", 
            "MV2_HOMOGENEOUS_CLUSTER=1",
            f'PYTHONPATH="{pypath}"',
            f"LD_PRELOAD={preload}"]
            if pyubase:
                cmd += [ f'PYTHONUSERBASE="{pyubase}"' ]
            # "MV2_USE_GDRCOPY=0"
            # "MV2_USE_RDMA_CM=0", "MV2_IBA_HCA=mlx5_0", "MV2_NUM_HCAS=1",
        print(f"mv2 cmd: {cmd}", flush=True)
    else:
        # This option enables proper crash behavior with spawn.
        # disconnected children don't crash the system.
        cmd += ["-mca", "orte_enable_recovery", "1"]
        #--mca btl ^vader,tcp,openib,uct  - disable tcp too?
        # cmd += [ "-mca", "pml", "ucx", "-mca", "btl", "^vader,uct,openib", "-x", "UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1" ]
        cmd += [ "-mca", "pml", "^ucx" ]
        cmd += ["-npernode", "1"]  # one jm_master per node
        cmd += ["-x", "PATH"]
        cmd += ["-x", "LD_LIBRARY_PATH"]
        cmd += ["-x", "PYTHONPATH"]
        # Need an external config file for these options
        cmd += ["-x", "JM_EOS_SET_AFFINITY"]
        cmd += ["-x", "JM_LAT_STARTUP_YAML"]
        cmd += ["-x", "EOS_NNEFFPWBASEPATH"]
        cmd += ["-x", "EOS_NNPWBASEPATH"]
        cmd += ["-x", "EOS_PARAMSDIR"]
    cmd += [jm_master_path]
    if jm_master_v:
        cmd += [jm_master_v]  # -v is for overly verbose output from jm_master
    if numlumps > 1:
        if i == 0:
            cmd += ["-blocks", "-py"] +  pyargs
            # add on to pyargs
            if blocksize > 0:
                cmd += ["-blocksize", str(blocksize)]
            if alloctime > 0:
                cmd += ["-alloctime", str(alloctime)]
        else:
            cmd += ["-connect"]
    else:
        cmd += ["-py"] + pyargs
        # add on to pyargs
        if blocksize > 0:
            cmd += ["-blocksize", str(blocksize)]
        if alloctime > 0:
            cmd += ["-alloctime", str(alloctime)]
    print(str(i)+": Launch", cmd, flush=True)
    logfilestr=logfilename(i)
    logfile=open(logfilestr, "w")
    x = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT )
    logfile.close()
    # Can use x.poll() to check for completion.    return is None if still running.    rc if done
    return [i, x, logfilestr]

# Tell the scheduler that it is time to take the nodes that have connected and begin running jobs
def launchend(host):
    # run on a single node.  Will exit.
    cmd = mpirun + ["-hosts", host, "-n", "1"] + mpirun_ns + [ "jm_master", "-v", "-end"]
    # cmd = mpirun + ["-np", "1", host] + mpirun_ns + [ "jm_master", "-v", "-end"]
    print("Send -end to stop collection of lumps, run on host ", host, flush=True)
    print("cmd=", cmd, flush=True)
    logfilestr="end.log"
    logfile=open(logfilestr, "w")
    x = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT )
    logfile.close()
    while x.poll is None:
        time.sleep(1)
        print("Waiting for -end launch to complete", flush=True)
    print("Signal -end to scheduler delivered", flush=True)

# Launch the name server that is used to make connections between mpirun's
def launchnameserver():
    global mpirun_ns
    print("Starting nameserver: ", nameserver, flush=True)
    logfile=open(nameserverlogfile, "w")
    x = subprocess.Popen(nameserver, stdout=logfile, stderr=subprocess.STDOUT )
    logfile.close()
    # the nameserver should keep running until the end of the job
    time.sleep(1) # wait to give it a chance to die
    if not (x.poll() is None):
        print("Nameserver ", nameserver, " failed to start, aborting ...", flush=True)
        exit(1)
    if not (nameserveridfile is None):
        mpirun_ns = ["--ompi-server", "file:" + nameserveridfile]
    return x  # return so we can terminate when done

def launchmpiruns(hostlist, numlumps, lumpsize, blocksize, alloctime, jm_master_v, pyargs):
    print("Starting ", numlumps, " lumps of nodes of size ", lumpsize, " from service node ", servicenode, flush=True)
    if blocksize > 0:
        print("blocksize set in command line to ", blocksize, flush=True)
    if alloctime > 0:
        print("alloctime set in command line to ", alloctime, flush=True)
    # write out host files for each of the runs
    for i in range(numlumps):
        lump = hostlist[i*lumpsize:(i+1)*lumpsize]
        writelump(i, lump)
    # The first lump with start the scheduler
    runlist = []
    print("Sleeping for 5 after launch of lump with scheduler", flush=True)
    for i in range(numlumps):
        if i == 1:
            time.sleep(5)
        if launchdelay:
            time.sleep(launchdelay)
        runlist.append(launchlump(i, numlumps, lumpsize, blocksize, alloctime, jm_master_v, pyargs))
    # Send end to finish collecting lumps.   Any stalled lumps will be left out
    time.sleep(enddelay)
    if numlumps > 1:
        # launchend(hostlist[0]) Used to run on compute node - may still have to for ssh reasons
        launchend(servicenode)  # run on service node.  Exits quickly anyway
    return runlist

def check_pid(pid):
    """
    Check for the existence of a Unix/Linux process with a given PID.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0) # kill 0 only tests if pid is active
    except OSError as e:
        if e.errno == errno.ESRCH:
            # ESRCH means "No such process"
            return False
        elif e.errno == errno.EPERM:
            # EPERM means "Permission denied" (process exists but we can't signal it)
            return True
        else:
            # Other errors might occur, like EINVAL (invalid signal)
            return False
    else:
        # No exception means the process is running and we have permission
        return True

def main():
    print("Start Mpi_Jm:  startup for multiple lumps of nodes using MVAPICH2-2.3.7", flush=True)
    s = datetime.datetime.now().isoformat()
    print("Start at time: ", s, flush=True)
    print("Servicenode: ", servicenode, flush=True)
    hostfile,badhosts,lumpsize,blocksize,alloctime,jm_master_v,pyargs=parseargs()
    hostlist = readhosts(hostfile, badhosts)
    hlen = len(hostlist)
    if hlen % lumpsize != 0:
        print("Trimming hostlist to multiple of lumpsize", flush=True)
        hostlist=hostlist[hlen%lumpsize:]
        hlen = len(hostlist)
        print(f"Trimmed hostlist has {hlen} nodes", flush=True)
    numlumps = len(hostlist) // lumpsize
    print("hosts=", hostlist, flush=True)

    # set up name server which enables connections between mpiruns
    if numlumps > 1:
        nameserver=launchnameserver()
    else:
        nameserver=None

    # start up the lumps.  The first one will include the scheduler
    runlist = launchmpiruns(hostlist, numlumps, lumpsize, blocksize, alloctime, jm_master_v, pyargs)

    print(f"There are {len(runlist)} lumps", flush=True)
    for r in runlist:
        i,x,fname = r
        print(f"  lump with mpirun pid {x.pid}", flush=True)

    pollpass=1
    donelist = []
    while runlist:
        time.sleep(5)
        print(f"polling pass {pollpass}, {len(runlist)} lumps remaining", flush=True)
        pollpass += 1
        nxtrunlist = []
        for r in runlist:
            i,x,fname = r
            rc = x.poll()  # test if process is still running
            print(f"  {i}: poll={rc}, pid={x.pid}", flush=True)
            if rc is None:
                if check_pid(x.pid):
                    nxtrunlist.append(r)
                else:
                    print(f"poll returned None but pid {x.pid} is gone for lump {i}!", flush=True)
                    donelist.append([i,x.pid,rc])
            else:
                donelist.append([i,x.pid,rc])
        runlist = nxtrunlist
    print("Start Mpi_Jm: All processes completed", flush=True)
    s = datetime.datetime.now().isoformat()
    print("Begin running jobs at Time: ", s, flush=True)
    for r in donelist:
        i,pid,rc=r
        print(str(i)+": pid=", pid, ", rc=", rc, flush=True)
    # shut down the name server before we go
    if not (nameserver is None):
        nameserver.terminate()

if __name__ == "__main__":
    # execute only if run as a script
    pickmpi() # configure for chosen MPI: openmpi or mvapich2
    main()
