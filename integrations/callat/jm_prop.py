import sys
import mpi_jm
import jm_common
import pathlib

# GLOBAL VARIABLES
executables=None# this is a global dictionary storing executables
quda_resource=None #default path for quda_resource

lalibe_pgm = '/usr/workspace/coldqcd/software/lassen_smpi_RR/install/lalibe_production_gpu/bin/lalibe'

def startcmd(j):
    print("Starting ", j.name)
    return jm_common.claimJob(j) # try to make <jobfile>.lock file

def wrapcmd(j):
    print("Wrapping ", j.name)

node_resources = {
    'lassen':["cpu|gpu", 4, 4],#cpu|gpu resource, number of ranks, number of threads/rank
    'darwin':["cpu", 1, 1],
}

ensemble_resources = {
    'a06m310L':{
        'lassen':[
            {'nodes':8,'geom':'1 1 2 16', 'OMP_NUM_THREADS':4, 'qmp-geom':'1 1 2 16', 'qmp-alloc-map':'3 2 1 0', 'qmp-logic-map':'3 2 1 0'},
            {'nodes':16,'geom':'1 2 2 16', 'OMP_NUM_THREADS':4,'qmp-geom':'1 2 2 16','qmp-alloc-map':'3 2 1 0','qmp-logic-map':'3 2 1 0'},
            ],
        'summit':[]
    }
}

def init_mod(cfg):
    global executables, quda_resource
    print("One time init for module")
    cx = cfg["executables"]
    # Fix paths, expanding $REL and ~user
    executables = {}
    for key,val in cx.items():
        executables[key] = jm_common.fixExePath(val)
    print(executables)
    # make sure tunecache dir exists ...
    quda_resource = cfg["quda_resource"]
    print('quda_resource =',quda_resource)

def add_job(jdict):
    print("Found way to prop.add_job!")
    print(jdict)
    j = mpi_jm.job() # create a new job
    j.wdir = jdict["run_dir"]
    j.name = jdict["yamlpath"]
    j.pgm = executables['opt_gpu']
    if 'executable' in jdict:
        try:
            j.pgm = executables[jdict['run_options']['executable']]
        except:
            jm_common.addJobFailed(jdict,'user specified executable key unknown: '+str(jdict['run_options']['executable']))
            return None
    cpugpu,nc,ng = node_resources[jm_common.machine]
    j.setresources(cpugpu, nc, ng)
    j.startcmd = "jm_prop.startcmd"
    j.wrapcmd = "jm_prop.wrapcmd"
    if 'quda_resource' in jdict['run_options']:
        print('jdict quda_resource',jdict['run_options']['quda_resource'])
        quda_resource = jdict['run_options']['quda_resource']
    j.addenv("QUDA_RESOURCE_PATH", j.wdir +'/'+ quda_resource)
    pathlib.Path(j.wdir +'/'+ quda_resource).mkdir(mode=0o770,parents=True,exist_ok=True)
    j.priority = jdict['priority']
    # get nnodes and geometry
    if 'nodes' in jdict['run_options']:
        eres = None
        for l in ensemble_resources[jdict['ensemble']][jm_common.machine]:
            print(l)
            print(l['nodes'])
            if l['nodes'] == jdict['run_options']['nodes']:
                eres = l
        if eres == None:
            jm_common.addJobFailed(jdict,'unsupported number of nodes: '+str(jdict['run_options']['nodes']))
            return None
    else:
        eres = ensemble_resources[jdict['ensemble']][jdict['machine']]
    nodes = eres['nodes']
    geom  = eres['geom']
    omp_num_thread = eres['OMP_NUM_THREADS']
    j.addenv("OMP_NUM_THREADS", omp_num_thread)
    j.addarg('-geom')
    j.addarg(geom)
    for q in ['qmp-geom','qmp-alloc-map','qmp-logic-map']:
        if q in eres:
            j.addarg('-'+q)
            j.addarg(eres[q])
    xml_in  = j.wdir + jdict['in_file']
    xml_out = j.wdir + jdict['out_file']
    stdout  = j.wdir + jdict['stdout']
    j.addarg('-i')
    j.addarg(xml_in)
    j.addarg('-o')
    j.addarg(xml_out)
    j.queue()
    return j

def add_job_og(jdict):
    print("Found way to prop.add_job!")
    print(jdict)
    j = mpi_jm.job() # create a new job
    j.wdir = jdict["run_dir"]+'/'+jdict["ensemble"]+'_'+jdict["stream"]
    j.name = jdict["yamlpath"]
    j.pgm = lalibe_pgm
    j.setresources("cpu|gpu", 4, 4)
    j.startcmd = "jm_prop.startcmd"
    j.wrapcmd = "jm_prop.wrapcmd"
    j.addenv("QUDA_RESOURCE_PATH", quda_res_path)
    # now stuff from the dictionary
    j.priority = jdict['priority']
    j.jobfile = jdict['jobfile']
    xml_in  = "xml/"+jdict["cfg"]+'/'+jdict["type"]+'_'+jdict["ensemble"]+'_'+jdict["stream"]+'_'+jdict["val_info"]+'_'+jdict["src"]+".ini.xml"
    xml_out = xml_in.replace('ini.xml','out.xml')
    stdout  = xml_in.replace('/xml/','/stdout/').replace('.ini.xml','.stdout')
    j.addarg('-i')
    j.addarg(xml_in)
    j.addarg('-o')
    j.addarg(xml_out)
    j.queue()
    return j

def trialJobDict():
    job_dict = dict()
    job_dict["type"]     = "prop"
    job_dict["ensemble"] = "a06m310L"
    job_dict["stream"]   = "b"
    job_dict["val_info"] = "gf1.0_w3.5_n45_M51.0_L56_a1.5"
    job_dict["cfg"]      = "469"
    job_dict["src"]      = "x8y36z71t7"
    job_dict["run_dir"]  = "/p/gpfs1/walkloud/c51/x_files/project_2/production/"
    job_dict["jobfile"]  = "prop_a06m310L_b_469_gf1.0_w3.5_n45_M51.0_L56_a1.5_mq0.00617_x8y36z71t7.yaml"
    job_dict["priority"] = 1
    job_dict["startcmd"] = "startcmd"
    job_dict["wrapcmd"]  = "wrapcmd"
    return job_dict

def main():
    d = trialJobDict()
    jm_common.addJob(d)

def schedInit():
    main()
