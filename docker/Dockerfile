ARG CUDA_VERSION=11.1.1
ARG BASE_IMAGE_TAG=${CUDA_VERSION}-devel-ubuntu18.04
FROM nvidia/cuda:${BASE_IMAGE_TAG}

# Install prerequisite apt packages
RUN apt-get update -q &&\
  apt-get install -yq \
  build-essential \
  curl \
  git \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install conda
ARG CONDA_VER=py39_4.10.3
ENV PATH="/opt/conda/bin:${PATH}"
RUN dpkg_arch="$(dpkg --print-architecture)"; \
  case "${dpkg_arch##*-}" in \
  amd64) conda_arch='x86_64';; \
  arm64) conda_arch='aarch64';; \
  ppc64el) conda_arch='ppc64le' ;; \
  s390x) conda_arch='s390x' ;; \
  *) echo "Unsupported architecture ${dpkg_arch}" > /dev/stderr; exit -1 ;; \
  esac; \
  curl -Ls https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VER}-Linux-${conda_arch}.sh -o /tmp/Miniconda.sh &&\
  bash /tmp/Miniconda.sh -p /opt/conda -b &&\
  rm /tmp/Miniconda.sh &&\
  conda update -y conda &&\
  conda clean -a &&\
  conda init

# Install pytorch, ogb, and pip
RUN conda install -y pytorch cudatoolkit=${CUDA_VERSION} -c pytorch-lts -c conda-forge &&\
  conda install -y -c conda-forge ogb &&\
  conda install -y -c anaconda pip &&\
  conda clean -a

# Build PyG dependencies for common GPU architectures, similar to how PyTorch is built.
# The builds are launched in parallel with shell background jobs.
# pytorch_sparse is installed in editable mode to enable us to undo upstream SALIENT changes
# for the performance comparisons.
RUN { export FORCE_CUDA=1 &&\
      export NVCC_FLAGS="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 $(dpkg --compare-versions ${CUDA_VERSION} gt 11 && echo -gencode arch=compute_80,code=sm_80 || echo) -gencode arch=compute_70,code=compute_70 -Xfatbin -compress-all"; };\
  /bin/bash -c '\
  pip install --upgrade-strategy only-if-needed git+git://github.com/rusty1s/pytorch_scatter.git@2.0.7 &\
  pip install --upgrade-strategy only-if-needed git+git://github.com/rusty1s/pytorch_cluster.git@1.5.9 &\
  pip install --upgrade-strategy only-if-needed git+git://github.com/rusty1s/pytorch_spline_conv.git@1.2.1 &\
  pip install --upgrade-strategy only-if-needed git+git://github.com/pyg-team/pytorch_geometric.git@1.7.0 &\
  { git clone https://github.com/rusty1s/pytorch_sparse /opt/pytorch_sparse && pip install --upgrade-strategy only-if-needed -e /opt/pytorch_sparse; } &\
  set -e && jobs && for job in $(jobs -p); do wait -n; done; wait' &&\
  { pip cache purge || echo "Nothing to remove"; }

# Install prettytable for artifact evaluation
RUN conda install -y -c conda-forge prettytable &&\
  conda clean -a
