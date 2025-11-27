import os
import sys
import mpi_jm
import jm_common
import pathlib

# GLOBAL VARIABLES
lalibe_executable_gpu=None # this is a global dictionary storing executables
lalibe_executable_cpu=None # this is a global dictionary storing executables
fixedargs=""
quda_resource=None #default path for quda_resource

# We have different lalibe jobs
subtypelist = ['prop', 'strangeprop', 'spec', 'source']
pulist = ['gpu', 'cpu']

# lalibe_pgm = '/usr/workspace/coldqcd/software/lassen_smpi_RR/install/lalibe_production_gpu/bin/lalibe'

def startcmd(j):
    print(f"Starting job {j.name}", flush=True)
    return jm_common.claimJob(j) # try to make <jobfile>.lock file

def create_prop_job_from_source(j):
    """
    When a source job completes, we generate a new
    prop job.
    """
    # First we create the name of the job and see the corresponding
    # yaml file exists.
    printf(f"Todo: Create new prop job from {j.name}")

def wrapcmd(j):
    subtype = j.dict['subtype']
    print(f"Wrapping {subtype} job {j.name}", flush=True)
    if subtype == 'source':
        # Looks like job is suppose to create new jobs
        create_prop_job_from_source(j)
        printf(f"Todo: Create new strangeprop job", flush=True)

def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

def init_mod(cfg):
    global lalibe_executable_cpu, lalibe_executable_gpu
    global fixedargs, quda_resource
    print("One time init for module jm_lalibe", flush=True)
    lalibe_executable_cpu = cfg.get("lalibe_executable_cpu", None)
    lalibe_executable_gpu = cfg.get("lalibe_executable_gpu", None)
    chroma_executable = cfg.get("chroma_executable", None)
    if lalibe_executable_cpu is None:
        raise ValueError(f"Missing path for jm_lalibe:lalibe_cpu executable");
    if lalibe_executable_gpu is None:
        raise ValueError(f"Missing path for jm_lalibe:lalibe_gpu executable");
    if chroma_executable is None:
        raise ValueError(f"Missing path for jm_lalibe:chroma_executable");

    lalibe_executable_cpu=lalibe_executable_cpu.format(**os.environ)
    lalibe_executable_gpu=lalibe_executable_gpu.format(**os.environ)
    lalibe_executable_cpu=jm_common.fixRelPath(lalibe_executable_cpu)
    lalibe_executable_gpu=jm_common.fixRelPath(lalibe_executable_gpu)
    print(f"jm_lalibe lalibe_executable_cpu: {lalibe_executable_cpu}")
    print(f"jm_lalibe lalibe_executable_gpu: {lalibe_executable_gpu}", flush=True)
    if not is_exe(lalibe_executable_cpu):
        raise ValueError(f"jm_lalibe lalibe_executable_cpu {lalibe_executable_cpu} isn't")
    if not is_exe(lalibe_executable_gpu):
        raise ValueError(f"jm_lalibe lalibe_executable_gpu {lalibe_executable_gpu} isn't")

    chroma_executable=chroma_executable.format(**os.environ)
    chroma_executable=jm_common.fixRelPath(chroma_executable)
    print(f"jm_lalibe chroma_executable: {chroma_executable}", flush=True)
    if not is_exe(chroma_executable):
        raise ValueError(f"jm_lalibe chroma_executable {chroma_executable} isn't")

    fixedargs = cfg.get('fixedargs', '').strip()
    if fixedargs == "":
        fixedargs = []
    else:
        fixedargs = fixedargs.split(' ')
    print(f"Propagator fixed args {fixedargs}", flush=True)
    quda_resource = os.environ['QUDA_RESOURCE_PATH'] # set in config
    if os.path.isdir(quda_resource):
        print(f"Found QUDA resource directory {quda_resource}")
    else:
        raise ValueError("QUDA resource directory {quda_resource} is missing")

def add_job(jdict):
    print(jdict)
    j = mpi_jm.job() # create a new job
    jobfile = jdict["yamlpath"].as_posix()
    # name can be shorter than jobfile, but must
    # uniquely specify the job
    j.jobfile = jobfile # path to job yaml file - used for locking
    logfile = jdict.get("stdout", "")
    if logfile != "":
        j.logfile = logfile
    j.name = jobfile    # can be shorter than jobfile - for reporting
    j.wdir = jdict["odir"].as_posix()
    pu = jdict.get('pu', '<missing>')  # either 'gpu' or 'cpu'
    if not (pu in pulist):
        raise ValueError(f"processing unit invalid, got {pu}")
    exe = lalibe_executable_cpu if pu == 'cpu' else lalibe_executable_gpu
    j.pgm = exe
    j.startcmd = "jm_lalibe.startcmd"
    j.wrapcmd = "jm_lalibe.wrapcmd"
    # j.addenv("QUDA_RESOURCE_PATH", quda_resource)
    j.addenv("PGPU_NOISY", "1")
    for a,b in jm_common.setenv_list:
        j.addenv(a, b)
    # Do we want to autocreate tuning directory?
    # This isn't the place.   We could use the first odir
    # pathlib.Path(j.wdir +'/'+ quda_resource).mkdir(mode=0o770,parents=True,exist_ok=True)
    j.priority = jdict.get('priority', 0) # support priority spec
    d = dict() # for extra parameters
    # 'prop', 'strangeprop', 'spec'
    subtype = jdict.get('subtype', '<missing>')
    if not (subtype in subtypelist):
        raise ValueError(f"Missing valid subtype, got {subtype}")
    d['subtype'] = jdict.get('subtype', None)
    d['pu'] = pu

    nnodes = jdict.get('nnodes', 4) # allow request of more/less nodes
    # -r1   number of resource sets/node
    # -a4   4 mpi ranks/resource set
    # -g4   4 gpus /resource set
    # -c4   4 cpus /resource set  (bigger than a4 if threads > core threads (4))
    # -nrs  total number of resource sets
    # gpu-cpu  optimize link between cpu's and gpus
    # packed:smt:4   pack ranks together with smt:4 (4 threads)
    # jsrun --nrs 6 -r1 -a4 -g4 -c4 -l gpu-cpu -b packed:smt:4 $PROG -i $ini -o $out 
    j.nodeMemKb = mpi_jm.getnodemem() // 100  # use small value
    # will pair cpu/gpu, putting 4 ranks on each node
    # Need to get this info from mpi_jm node info
    nthreads = 4 # 4 threads/rank - 1 cores with smt4
    gpuspernode = 4
    if pu == 'gpu':
        # Need to rework setresources - want one cpu|gpu slot, and 3 more cpu slots for 4 threads
        # for now, use OMP_NUM_THREADS=4
        j.setresources("cpu|gpu", nnodes*gpuspernode, nthreads)
        # The problem is that we want CPU with 4 slots/threads
        # but one gpu slot.
        j.priority=10
        # add args shared by all jobs
        for fa in fixedargs:
            j.addarg(fa)
    else:
        # If we want to make it possible to overlay cpu jobs, then
        # gpucores = 2/socket
        # we have (nodes/socket - coreisolation/socket - gpuspersocket)*2 = (22 - 2 - 2)*2 = 36
        j.setresources("cpu", nnodes*36, nthreads)
        j.priority=1
    j.addenv("OMP_NUM_THREADS", str(nthreads))
    xml_in  = jdict['xml_in']
    # Lets make an xml_out from the xml_in
    # It will be local to the output directory
    xml_out = jdict.get('xml_out', None)
    if xml_out is None:
        xml_out = xml_in.rsplit('/', 1)[-1].replace('.ini.', '.out.')
    # stdout is already handled.
    j.addarg('-i')
    j.addarg(xml_in)
    j.addarg('-o')
    j.addarg(xml_out)
    j.dict = d # save extra parameter dictionary on j
    j.queue()
    return j
