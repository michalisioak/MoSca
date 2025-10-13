NUMPY_VERSION=1.26.4

python3.10 -m venv .venv
source .venv/bin/activate

pip install numpy==$NUMPY_VERSION
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install fvcore iopath

pip install nvidia-pip  # if required for CUB headers
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install pyg_lib torch_scatter torch_geometric torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install xformers
pip install -r requirements.txt
pip install numpy==$NUMPY_VERSION


pip install lib_render/simple-knn
pip install lib_render/diff-gaussian-rasterization-alphadep-add3
pip install lib_render/diff-gaussian-rasterization-alphadep
pip install lib_render/gof-diff-gaussian-rasterization

pip install numpy==$NUMPY_VERSION
pip install -U scikit-learn 
pip install -U scipy
pip install opencv-python==4.10.0.84