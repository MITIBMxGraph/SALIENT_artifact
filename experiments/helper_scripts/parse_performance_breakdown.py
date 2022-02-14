import argparse
import operator
from argparse import Namespace
import os
import statistics
import prettytable

parser = argparse.ArgumentParser(description="Parse SALIENT experiment logs")


parser.add_argument(
    "directory", help="Name of directory containing tests to parse.", type=str)


args = parser.parse_args()


def parse_logfile(filename):
    lines = open(filename).readlines()

    args_file = open(filename.replace("logs.txt", "args.txt")).read()
    experiment_args = eval(args_file)

    train_sampler = experiment_args.train_sampler
    if train_sampler == 'NeighborSampler':
        train_sampler = 'PyG'
    else:
        train_sampler = 'SALIENT'

    dataset = experiment_args.dataset_name

    row = []
    row.append(train_sampler)
    row.append(dataset)
    row.append(experiment_args.model_name)

    # Output format example: ('performance_breakdown_stats', "[['Data Transfer', 0.5243520042276941, 0.06822215054862966], ['Sampling + Slicing', 51.23573551449226, 1.5084828744159517], ['Total', 446.60448725521564, 1.5387180068033377], ['Train', 394.4309269785881, 0.8067731601368003]]")
    for line in lines:
        if line.startswith("('performance_breakdown_stats'"):
            info = dict()
            x = line.split('"')[1]
            stats = eval(x)
            for x in stats:
                info[x[0]] = "{:.3f}".format(
                    float(x[1])) + " " + u'\xb1' + " " + "{:.3f}".format(float(x[2]))
            row.append(info['Total'])
            row.append(info['Train'])
            row.append(info['Sampling + Slicing'])
            row.append(info['Data Transfer'])
            return row
            break
    return None


rows = []
for x in os.listdir(args.directory):
    if x.startswith("nodelist_"):
        continue

    subdir = os.path.join(args.directory, x)
    print("Experiment at " + subdir)
    try:
        for y in os.listdir(subdir):
            if y.endswith("logs.txt"):
                info = parse_logfile(os.path.join(subdir, y))
                if info is None:
                    print(
                        "Experiment not complete, not adding results experiment directory: " + subdir)
                else:
                    rows.append(info)
    except:
        print("Error when processing one of the experiments")

tab = prettytable.PrettyTable()
tab.field_names = ["Salient/PyG", "Dataset", "Model",
                   "Total (ms)", "Train (ms)", "Sampling + Slicing (ms)", "Data Transfer (ms)"]
tab.hrules = prettytable.ALL
tab.float_format = ".3"
for r in rows:
    tab.add_row(r)
print(tab.get_string(sort_key=operator.itemgetter(0, 1), sortby="Dataset"))
