import argparse
from argparse import Namespace
import os
import statistics
import prettytable
import operator

parser = argparse.ArgumentParser(description="Parse SALIENT experiment logs")


parser.add_argument(
    "directory", help="Name of directory containing tests to parse.", type=str)


args = parser.parse_args()


def parse_logfile(filename):
    lines = open(filename).readlines()

    args_file = open(filename.replace("logs.txt", "args.txt")).read()
    experiment_args = eval(args_file)

    experiment_info_list = []
    train_sampler = experiment_args.train_sampler
    if train_sampler == 'NeighborSampler':
        train_sampler = 'PyG'
    else:
        train_sampler = 'SALIENT'
    experiment_info_list.append(str(experiment_args.model_name))
    experiment_info_list.append(str(train_sampler))
    experiment_info_list.append("Dataset:" + str(experiment_args.dataset_name))
    experiment_info_list.append(
        "GPUs-Per-Node:" + str(experiment_args.max_num_devices_per_node))
    experiment_info_list.append(
        "Nodes:" + str(experiment_args.total_num_nodes))
    experiment_info_list.append(
        "CPU per GPU:" + str(experiment_args.num_workers))
    experiment_info_list.append("Num epochs:" + str(experiment_args.epochs))
    experiment_info_list.append("Num trials:" + str(experiment_args.trials))

    valid_timer_lines = []
    test_timer_lines = []
    train_timer_lines = []
    result_list = None
    for x in lines:
        if x.startswith("TimerResult"):
            if x.find("'valid'") != -1:
                valid_timer_lines.append(x)
                continue
            if x.find("'test'") != -1:
                test_timer_lines.append(x)
                continue
            # otherwise it is a training timer.
            # ignore the first epoch
            if x.find("name=(0") == -1:
                train_timer_lines.append(x)
        if x.startswith("('End results for all trials',"):
            text = x.replace("('End results for all trials',", "")
            result_list = eval(text.replace("'", "").strip()[:-1])

    epoch = 1
    train_compute_times = []
    train_preamble_times = []
    for x in train_timer_lines:
        if x.find("name=("+str(epoch)+", 'Preamble')") != -1:
            train_preamble_times.append(
                int(str(x.split("nanos=")[1].replace(")", ""))))

        if x.find("name=("+str(epoch)+", 'Compute')") != -1:
            train_compute_times.append(
                int(str(x.split("nanos=")[1].replace(")", ""))))
            epoch += 1

    assert len(train_compute_times) == experiment_args.epochs-1 and \
        len(train_preamble_times) == experiment_args.epochs-1

    total_train_times = []
    for i in range(0, len(train_compute_times)):
        total_train_times.append(
            (1.0*(train_compute_times[i] + train_preamble_times[i]))/1e+9)

    return (statistics.mean(total_train_times), statistics.stdev(total_train_times), result_list, experiment_info_list)

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
                print(info)
                if info[2] != None:
                    accuracies = info[2]
                    valid_acc_list = []
                    test_acc_list = []
                    row = [info[3][0], info[3][1], "\n".join(
                        info[3][2:]), "{:.3f}".format(info[0]) + " " + u'\xb1' + " " + "{:.3f}".format(info[1])]
                    for acc in accuracies:
                        valid_acc_list.append(acc[0])
                        test_acc_list.append(acc[1])

                    if len(valid_acc_list) > 1:
                        row.append("{:.3f}".format(statistics.mean(valid_acc_list)) +" " + u'\xb1' + " " + "{:.3f}".format(statistics.stdev(valid_acc_list)))
                    else:
                        row.append("{:.3f}".format(statistics.mean(valid_acc_list)) +" " + u'\xb1' + " " + "N/A")

                    if len(test_acc_list) > 1:
                        row.append("{:.3f}".format(statistics.mean(test_acc_list)) + " " + u'\xb1' + " " + "{:.3f}".format(statistics.stdev(test_acc_list)))
                    else:
                        row.append("{:.3f}".format(statistics.mean(test_acc_list)) + " " + u'\xb1' + " " + "N/A")
                    rows.append(row)
    except:
        print("Error")

tab = prettytable.PrettyTable()
tab.field_names = ["Model", "System", "Params",
                   "Epoch time (s)", "Valid acc", "Test acc"]
tab.hrules = prettytable.ALL
tab.float_format = ".3"
for r in rows:
    tab.add_row(r)
print(tab.get_string(sort_key=operator.itemgetter(1,0), sortby="System"))
