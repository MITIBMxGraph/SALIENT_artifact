import ogb
from ogb.utils.url import decide_download, download_url, extract_zip
import os


directory = os.path.dirname(os.path.abspath(__file__))
dataset_dir = directory + "/dataset"


dataset_base_url = "https://salient-datasets-ae.s3.amazonaws.com/"

#dataset_list = ["https://salient-datasets-ae.s3.amazonaws.com/ogbn-arxiv.zip","https://salient-datasets-ae.s3.amazonaws.com/ogbn-products.zip"]


if not os.path.exists(directory + "/.feasible_datasets"):
    print("[Error] Did not detect list of feasible datasets. Please run experiments/configure_for_environment.py before trying to download datasets.")
    quit()


datasets = open(directory + "/.feasible_datasets").readlines()
dataset_list = [x.strip() for x in datasets]

infeasible_datasets = open(directory + "/.infeasible_datasets").readlines()
infeasible_dataset_list = [x.strip() for x in infeasible_datasets]

message = """\
###############################################################################
######## Downloading preprocessed OGB datasets for artifact evaluation ########
###############################################################################
"""
print()
print(message)

print()
print("[Info] Will try to download: " + str(",".join(dataset_list)))
print("[Info] Will *not* try to download: " +
      str(",".join(infeasible_dataset_list)))
print()


for dataset in dataset_list:
    print()
    print("Trying to download " + dataset)
    dataset_name = dataset
    #print (dataset_dir + "/"+dataset_name)
    if os.path.exists(dataset_dir + "/" + dataset_name):
        print("Skip " + dataset + " because it already exists.")
    else:
        dataset_url = dataset_base_url + dataset + ".zip"
        if decide_download(dataset_url):
            path = download_url(dataset_url, dataset_dir)
            extract_zip(path, dataset_dir)
            os.unlink(path)
    print()
