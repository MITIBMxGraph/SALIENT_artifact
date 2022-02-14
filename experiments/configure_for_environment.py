import logging
import subprocess
import sys
import os
import time
import csv
import shutil
logger = logging.getLogger(__name__)


def run_command(cmd, asyn=False):
    proc = subprocess.Popen(
        [cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not asyn:
        out, err = proc.communicate()
        return out, err
    else:
        return ""


def get_n_cpus():
    return len(get_cpu_ordering())


def get_cpu_ordering():
    if sys.platform == "darwin":
        # TODO: Replace with something that analyzes CPU configuration on Darwin
        out, err = run_command("sysctl -n hw.physicalcpu_max")
        return [(0, p) for p in range(0, int(str(out, 'utf-8')))]
    else:
        out, err = run_command("lscpu --parse")

    out = str(out, 'utf-8')

    avail_cpus = []
    for l in out.splitlines():
        if l.startswith('#'):
            continue
        items = l.strip().split(',')
        cpu_id = int(items[0])
        core_id = int(items[1])
        socket_id = int(items[2])
        avail_cpus.append((socket_id, core_id, cpu_id))

    avail_cpus = sorted(avail_cpus)
    ret = []
    added_cores = dict()
    for x in avail_cpus:
        if x[1] not in added_cores:
            added_cores[x[1]] = True
        else:
            continue
        ret.append((x[2], x[0]))
    return ret


dir_path = os.path.dirname(os.path.realpath(__file__))


def adapt_config_files():

    message = """\
###############################################################################
######## Modifying experiment configuration files to adapt to hardware ########
###############################################################################
"""
    print(message)

    # determine number of physical CPUs
    try:
        cpu_ordering = get_cpu_ordering()
        num_physical_cpus = len(cpu_ordering)
        hyperthreads_per_core = len(cpu_ordering[0])

        desired_workers = 30

        print("Configuration of number of sampling worker threads")
        print("\t[INFO] Detected " + str(num_physical_cpus) +
              " physical CPUs each with " + str(hyperthreads_per_core) + " hardware threads.")
        if num_physical_cpus >= 30 and hyperthreads_per_core >= 2:
            desired_workers = 30
        elif num_physical_cpus >= 20 and hyperthreads_per_core >= 2:
            desired_workers = num_physical_cpus + num_physical_cpus/2
        elif num_physical_cpus >= 20 and hyperthreads_per_core == 1:
            desired_workers = num_physical_cpus-1
            #print("[INFO] ")
        elif hyperthreads_per_core >= 2:
            desired_workers = num_physical_cpus + num_physical_cpus/2
            print("\t[WARNING] Detected fewer than 20 physical CPUs.")
        elif hyperthreads_per_core == 1:
            desired_workers = num_physical_cpus
            print(
                "\t[WARNING] Detected fewer than 20 physical cpus and no hyperthreading. Performance likely to be suboptimal.")
        desired_workers = int(desired_workers)
        print("\t[INFO] Setting desired workers to " +
              str(desired_workers) + ".")
    except:
        print("Error when analyzing hardware system to adapt config files")
        exit(1)

    update_configurations = ['performance_breakdown_config.cfg']

    for cfg in update_configurations:
        lines = open(
            dir_path + '/performance_breakdown_config.cfg').readlines()
        new_lines = []
        for l in lines:
            l = l.strip()
            if l.startswith('--num_workers'):
                new_lines.append('--num_workers ' + str(desired_workers))
            else:
                new_lines.append(l)
        open(dir_path + '/performance_breakdown_config.cfg',
             'w').write("\n".join(new_lines))
        print("*Updated configuration for file " + cfg)

    open(dir_path + '/.ran_config', 'w+').write("1")


def dataset_feasibility(free_gb):
    dataset_list = []
    dataset = ('ogbn-arxiv', 0.5)
    dataset_list.append(dataset)
    dataset = ('ogbn-products', 2.0)
    dataset_list.append(dataset)
    dataset = ('ogbn-papers100M', 100.0)
    dataset_list.append(dataset)
    infeasible = []
    feasible = []
    for x in dataset_list:
        if x[1] <= free_gb:
            feasible.append(x)  # x[0] + " ("+str(x[1])+" GB needed)")
        else:
            infeasible.append(x)  # x[0] + " ("+str(x[1])+" GB needed)")
    return feasible, infeasible


def determine_viable_datasets():
    import shutil

    message = """\
###############################################################################
######## Checking disk space to decide datasets to use for experiments ########
###############################################################################
"""
    print()
    print(message)

    total, used, free = shutil.disk_usage(__file__)
    free_gb = (1.0*free)/1e+9
    feasible, infeasible = dataset_feasibility(free_gb)
    print("\t[INFO] Checking disk space available in directory " + dir_path)
    print("\t[INFO] Available disk space detected: " + str(free_gb) + " GB")
    if len(feasible) == 0:
        print("\t[Warning] **Extremely** low disk space available. You may not be able to download any datasets. Please free space before continuing.")
    if len(feasible) == 1:
        print("\t[Warning] Very low disk space. It is strongly recommended to free additional space. Otherwise you may only run on the smallest datasets.")
    if len(feasible) == 2:
        print("\t[Warning] Somewhat low disk space. You can run on small/medium sized datasets. Free space to run on larger datasets.")
    print("\t[Info] Sufficient space for datasets: " +
          "; ".join([x[0] + " ("+str(x[1])+" GB needed)" for x in feasible]))
    print("\t[Info] Insufficient space for datasets: " +
          "; ".join([x[0] + " ("+str(x[1])+" GB needed)" for x in infeasible]))

    dataset_dir = dir_path+"/dataset"
    for x in infeasible:
        if os.path.exists(dataset_dir + "/" + x[0]):
            print("\t[Info] Dataset " + x[0] +
                  " exists at " + dataset_dir + "/" + x[0])
            print(
                "\t\tWe thought this dataset might be too big, but it seems you've already downloaded it.")
            feasible.append(x)
    for x in feasible:
        if x in infeasible:
            infeasible.remove(x)
        print("\t[Info] Dataset " + x[0] + " will be recorded as feasible")

    open(dir_path+"/.feasible_datasets",
         "w").write("\n".join([str(x[0]) for x in feasible]))
    open(dir_path+"/.infeasible_datasets",
         "w").write("\n".join([str(x[0]) for x in infeasible]))
    print("\t[Info] Updated experiments/.feasible_datasets and experiments/.infeasible_datasets")

    feasible_list_str = "\n".join(
        ["\t\t-" + x[0] + " ("+str(x[1])+" GB needed)" for x in feasible])
    infeasible_list_str = "\n".join(
        ["\t\t-" + x[0] + " ("+str(x[1])+" GB needed)" for x in infeasible])

    print(f"""
    You can run on the datasets:
    {feasible_list_str}

    You cannot run on the datasets:
    {infeasible_list_str}


    To download the datasets we detected as feasible for your available disk space

        Run: python download_datasets_fast.py

    If you are running via the experiments/initial_setup.sh script, then datasets will be downloaded after this script.

    """)


adapt_config_files()
determine_viable_datasets()


print("Done")
exit(0)
