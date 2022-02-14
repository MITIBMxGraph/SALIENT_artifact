# Setup Instructions on a GPU Machine

### 1. Install Conda

Follow instructions on the [Conda user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). For example, to install Miniconda on an x86 Linux machine:

```bash
curl -Ls https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VER}-Linux-${conda_arch}.sh -o /tmp/Miniconda.sh &&\
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh
```

SALIENT has been tested on Python 3.9.5.

It is highly recommended to create a new environment and do the subsequent steps therein. Example with a new environment called `salient`:

```bash
conda create -n salient python=3.9.5 -y
conda activate salient
```

You will also need to install `pip` for the subsequent steps:

```bash
conda install -y -c anaconda pip
```

### 2. Install PyTorch

Follow instructions on the [PyTorch homepage](https://pytorch.org). For example, to install on a linux machine with CUDA 11.3:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

SALIENT was developed against PyTorch 1.8.1 and has been tested up to PyTorch 1.10.0.

### 3. Install OGB

```bash
conda install -y -c conda-forge ogb
```

SALIENT has been tested on OGB 1.3.2.

### 4. Install PyTorch-Geometric (PyG)

#### Option A: Latest PyG

To get the latest version of PyG, follow the instructions on the [PyG Github page](https://github.com/pyg-team/pytorch_geometric).

```bash
conda install pyg -c pyg -c conda-forge
```

#### Option B: Versions used in paper

To install the versions used in the paper, you must build from source.
Make sure you have the `nvcc` compiler installed as part of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
If you take this option, you can skip Step 5 below.

```bash
export FORCE_CUDA=1
pip install git+git://github.com/rusty1s/pytorch_scatter.git@2.0.7
pip install git+git://github.com/rusty1s/pytorch_cluster.git@1.5.9
pip install git+git://github.com/rusty1s/pytorch_spline_conv.git@1.2.1
pip install git+git://github.com/rusty1s/pytorch_sparse.git@master
pip install git+git://github.com/pyg-team/pytorch_geometric.git@1.7.0
```

SALIENT was developed against PyG 1.7.0 and has been tested up to PyG 2.0.3.

### 5. Install Modified PyTorch-Sparse

PyTorch-Sparse is usually installed together with PyG. We made a slight modification to support faster data transfer.
We recently contributed this [patch](https://github.com/rusty1s/pytorch_sparse/pull/195) to the upstream `pytorch_sparse` repo that, at the time of releasing this artifact, has been merged but not yet distributed.
Before a `pytorch_sparse` release with version greater than 0.6.12 is available, you will need to install this package from source before installing PyG.
If you followed option B in Step 4, you can skip the current step.

To install from source, you will need the `nvcc` compiler.
Uninstall the prior version if it exists and install the latest version from source:

```bash
pip uninstall torch_sparse
FORCE_CUDA=1
pip install git+git://github.com/rusty1s/pytorch_sparse.git@master
```

### 6. Install SALIENT's fast_sampler

Go to the `fast_sampler` folder and install it:

```bash
cd fast_sampler
python setup.py install
```

To check that it is properly installed, start python and type:

```python
>>> import torch
>>> import fast_sampler
>>> help(fast_sampler)
```

You should see information of the package.

Note: Compilation requires a C++ compiler that supports C++17 (e.g., gcc >= 7).

### 7. Install Other Dependencies

```bash
conda install prettytable -c conda-forge
```
