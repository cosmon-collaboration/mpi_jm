# vim: tabstop=4 expandtab
# Set up block structure and resources.
# The total number of nodes is passed by jm_master
# The number of blocks is numnodes/blocksize
import mpi_jm
import os
import sys
import pathlib
print("Loading jm_machine.py", flush=True);

gpus_per_node = 4  # will update below
cores_per_node = os.cpu_count() // 2

# These parameters are used for testing on desktop
def setMachineParametersDflt():
    global gpus_per_node, cores_per_node
    print("Setting Machine Parameters for Default")
    print("Defining node/block resources");
    sys.stdout.flush()
    mpi_jm.setnodemem(32*1024*1024) # Kb/node
    mpi_jm.setblocksize(1) # in nodes - for mac we get just one node
    mpi_jm.addnoderesource("cpu")
    mpi_jm.addnoderesource("gpu")
    for id in range(0,3):
        mpi_jm.addslot("cpu")
    mpi_jm.addslot("cpu|gpu")
    for id in range(4,7):
        mpi_jm.addslot("cpu")
    mpi_jm.addslot("cpu|gpu")

    mpi_jm.addslotenv(0, "EMOTION", "happy");
    mpi_jm.addslotenv(4, "EMOTION", "sad");
    gpus_per_node = 2
    mpi_jm.resfinish()

def setMachineParametersPerlmutter():
    global gpus_per_node, cores_per_node
    # could be running on gpu machine or cpu machine
    mpi_jm.addnoderesource("cpu")
    if cores_per_node == 128:
        print("Setting machine parameters for Perlmutter CPU node with {cores_per_node} cores")
        gpus_per_node = 0
        for id in range(cores_per_node):
            mpi_jm.addslot("cpu")
    elif cores_per_node == 64:
        print("Setting machine parameters for Perlmutter GPU with {cores_per_node} cores")
        gpus_per_node = 4
        mpi_jm.addnoderesource("gpu")
        for id in range(cores_per_node):
            if (id % 16) == 0:  # There will be 4 GPUs
                mpi_jm.addslot("cpu|gpu")
                # Pick the GPU for the rank starting here.
                # We will associate a rank on the node with a GPU, then OMP_NUM_THREADS=16
                mpi_jm.addslotenv(id, "CUDA_VISIBLE_DEVICES", str(id//16))  # 0,1,2,4
                mpi_jm.addslotenv(id, "PGPU_RANKDEV", "0") # This helps eos app, says rank uses dev 0, from above devices
            else:
                mpi_jm.addslot("cpu")
    else:
        raise ValueError(f"Error:  Perlmutter node has {cores_per_node} nodes, expecting 64(GPU node), 128(CPU node)")
    mpi_jm.resfinish()

def setMachineParametersKensho():
    global gpus_per_node, cores_per_node
    # Kensho has 20 cores, 40 "cpus" and 4 GPUs, 256Gb ram
    print("Setting machine parameters for Kensho")
    gpus_per_node = 4
    sys.stdout.flush()
    mpi_jm.setnodemem(256*1024*1024) # Kb/node
    mpi_jm.setblocksize(1) # in nodes - for kensho we have one node
    mpi_jm.addnoderesource("cpu")
    mpi_jm.addnoderesource("gpu")
    for i in range(0,40,10):
        for id in range (i,i+9):
            mpi_jm.addslot("cpu")
        mpi_jm.addslot("cpu|gpu")
    mpi_jm.resfinish()

def setMachineParametersSummit():
    print("Setting Machine Parameters for Summit")
    gpus_per_node = 6
    sys.stdout.flush()
    mpi_jm.setnodemem(32*1024*1024) # Kb/node
    mpi_jm.setblocksize(4) # in nodes
    mpi_jm.addnoderesource("cpu")
    mpi_jm.addnoderesource("gpu")
    mpi_jm.addnoderesource("gpu2")
    mpi_jm.addnoderesource("gpu4")
    mpi_jm.addnoderesource("dummy")
    # adding NIC resource specifications to GPU slots
    # slide 6 of
    # https://www.olcf.ornl.gov/wp-content/uploads/2018/02/SummitJobLaunch.pdf
    # shows a Summit node
    # 21 CPU cores per node with 4 hardware threads per core
    # some core numbers are skipped
    # we are chosing to slot the 3 GPUs/NUMA-node to the first 3 CPU cores
    for id in range(0, 44*4):
        rs = "cpu"
        if id in [84,85,86,87,172,173,174,175]:
            # summit SW reserves these cores for OS functions
            mpi_jm.addslot("dummy")
        elif id in [0,8,16]:
            rs += "|gpu"
            if id == 0:
                rs += "|gpu2"
            else:
                rs += "|gpu4"
            mpi_jm.addslot(rs)
            # pick one GPU for this CPU slot
            mpi_jm.addslotenv(id, "CUDA_VISIBLE_DEVICES", str(id//8))  # 0,1,2
            mpi_jm.addslotenv(id, "PGPU_RANKDEV", "0")
            # FIXME:  need way to detect which mpi we are using
            # for OPENMPI
            # mpi_jm.addslotenv(id, "OMPI_MCA_btl_openib_if_include", "mlx5_1")
            # mpi_jm.addslotenv(id, "PAMI_IBV_DEVICE_NAME", "mlx5_0:1")
            # for MVAPICH
            # mpi_jm.addslotenv(id, "MV2_IBA_HCA", "mlx5_1:1")
        elif id in [88,96,104]:
            rs += "|gpu"
            if id == 88:
                rs += "|gpu2"
            else:
                rs += "|gpu4"
            mpi_jm.addslot(rs)
            mpi_jm.addslotenv(id, "CUDA_VISIBLE_DEVICES", str(3+(id-88)//8))  # 0,1,2
            mpi_jm.addslotenv(id, "PGPU_RANKDEV", "0")
            # for OPENMPI
            # mpi_jm.addslotenv(id, "OMPI_MCA_btl_openib_if_include", "mlx5_3")
            # mpi_jm.addslotenv(id, "PAMI_IBV_DEVICE_NAME", "mlx5_3:1")
            # for MVAPICH
            # mpi_jm.addslotenv(id, "MV2_IBA_HCA", "mlx5_3:1")
        else:
            mpi_jm.addslot(rs)
    mpi_jm.resfinish()

# From sierra documentation.
# Note that tasks do not use 0-7 or 88-95  2 cores/socket are reserved for "core-isolation"
# Basically, system tasks are confined to 2 cores on each socket and user ranks should 
# use the remaining cores to avoid interference.
#
# % lrun -n4 js_task_info
# Task 0 ( 0/4, 0/4 ) is bound to cpu[s] 8,12,16,20,24,28,32,36,40,44 on host lassen2 with OMP_NUM_THREADS=10 and with OMP_PLACES={8},{12},{16},{20},{24},{28},{32},{36},{40},{44} and CUDA_VISIBLE_DEVICES=0
# Task 1 ( 1/4, 1/4 ) is bound to cpu[s] 48,52,56,60,64,68,72,76,80,84 on host lassen2 with OMP_NUM_THREADS=10 and with OMP_PLACES={48},{52},{56},{60},{64},{68},{72},{76},{80},{84} and CUDA_VISIBLE_DEVICES=1
# Task 3 ( 3/4, 3/4 ) is bound to cpu[s] 136,140,144,148,152,156,160,164,168,172 on host lassen2 with OMP_NUM_THREADS=10 and with OMP_PLACES={136},{140},{144},{148},{152},{156},{160},{164},{168},{172} and CUDA_VISIBLE_DEVICES=3
# Task 2 ( 2/4, 2/4 ) is bound to cpu[s] 96,100,104,108,112,116,120,124,128,132 on host lassen2 with OMP_NUM_THREADS=10 and with OMP_PLACES={96},{100},{104},{108},{112},{116},{120},{124},{128},{132} and CUDA_VISIBLE_DEVICES=2
def setMachineParametersSierra():
    global gpus_per_node, cores_per_node
    print("Setting Machine Parameters for Sierra/Lassen")
    gpus_per_node = 4
    sys.stdout.flush()
    mpi_jm.setnodemem(256*1024*1024) # Kb/node
    mpi_jm.setblocksize(8) # in nodes
    mpi_jm.addnoderesource("cpu")
    mpi_jm.addnoderesource("gpu")
    mpi_jm.addnoderesource("sys")
    # Sierra lstopo says:
    #   There are 2 numa nodes with 128Gb each.
    #   Each numa node has 22 cores each of which can run 4 threads.
    #   The cores come in pairs that share a 512Kb data cache.
    #   Each core has it's own 32Kb instruction cache.
    #   Each numa node has 2 "render" GPUs that I think we ignore and two other GPU's that are probably the ones you want.
    #   That gives a total of 176 threads/node and 4 GPUs.
    for id in range(176):
        # would be [0, 44, 88, 136] except we skip over core isolation at beginning of sockets
        if id in [8, 44, 96, 136]:
            mpi_jm.addslot("cpu|gpu")
            gs = id // 44
            # mpi_jm.addslotenv(id, "CUDA_VISIBLE_DEVICES", str(gs))
            mlxn = 2*(id // 88) + 1   # either 1 or 3
            mlxs = "mlx5_" + str(mlxn)
            # mpi_jm.addslotenv(id, "OMPI_MCA_btl_openib_if_include", mlxs)
            # mpi_jm.addslotenv(id, "MV2_IBA_HCA", mlxs)
        elif id < 8 or (id >= 88 and id < 96):
            mpi_jm.addslot("sys")
        else:
            mpi_jm.addslot("cpu")
    mpi_jm.resfinish()


def setMachineParameters():
    # on Summit, with openMPI, env are not getting passed to compute nodes
    # so we manually set hostname to summit for now if using openMPI
    #hostname="summit"
    # hostname=socket.gethostname()  - Something wrong with import.
    hostname=os.uname()[1]
    print("Detecting machine:", hostname)
    sys.stdout.flush()

    clustername=os.getenv("SLURM_CLUSTER_NAME", 'none')

    if clustername == "perlmutter":
        setMachineParametersPerlmutter()
    elif pathlib.Path('/sw/summit').exists():
        setMachineParametersSummit()
    elif "kensho" in hostname:
        setMachineParametersKensho()
    elif "sierra" in hostname or "lassen" in hostname:
        setMachineParametersSierra()
    else:
        hostname=os.getenv('LSB_EXEC_CLUSTER', 'none')
        print('hostname:',hostname)
        sys.stdout.flush()
        if "summit" in hostname:
            setMachineParametersSummit()
        else:
            # macs go here
            setMachineParametersDflt()
