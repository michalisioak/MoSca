ENV_NAME=mosca
NUMPY_VERSION=1.26.4

conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME gcc_linux-64=9 gxx_linux-64=9 python=3.10 -y
conda activate $ENV_NAME
which python
which pip
# conda install nvidia/label/cuda-11.8.0::cuda-nvcc
which nvcc
CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
CPP=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
$CC --version
$CXX --version   
# conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c nvidia -c pytorch -y 
conda install numpy=$NUMPY_VERSION -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# pip install torch==2.1.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia 
conda install nvidiacub -c bottler -y
conda install pytorch3d -c pytorch3d -y
pip install pyg_lib torch_scatter torch_geometric torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
################################################################################

################################################################################
echo "Install other dependencies..."
# conda install xformers -c xformers -y
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install numpy==$NUMPY_VERSION
# pip install "typing-extensions>=4.14.1,<5" --upgrade
################################################################################
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
################################################################################
echo "Install GS..."
pip install lib_render/simple-knn
pip install lib_render/diff-gaussian-rasterization-alphadep-add3
pip install lib_render/diff-gaussian-rasterization-alphadep
pip install lib_render/gof-diff-gaussian-rasterization
################################################################################

################################################################################
pip install numpy==$NUMPY_VERSION
pip install -U scikit-learn 
pip install -U scipy
pip install opencv-python==4.10.0.84
pip install -U openmim
mim install "mmcv-full==1.7.2"
################################################################################

################################################################################
echo "Install JAX for evaluating DyCheck"
pip install -r jax_requirements.txt
################################################################################