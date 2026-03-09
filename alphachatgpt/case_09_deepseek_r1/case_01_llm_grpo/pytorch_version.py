import torch; 

"""
# conda clean --all
conda create -p py311_unsloth python=3.11 pytorch-cuda=12.1 pytorch=2.4.0 cudatoolkit -c pytorch -c nvidia -y
conda create -p 'py311_unsloth' \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch=2.4.0 cudatoolkit -c pytorch -c nvidia \
    -y
    # pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \ 

# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# pip uninstall torch torchvision torchaudio
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27.post2

# conda install pytorch==2.5.1 torchvision==0.20.0 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.0 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.28.post3

# xformers-0.0.27.post2-cp311-cp311-win_amd64.whl

# torch-2.5.0-cp311-cp311-win_amd64.whl

# numpy-2.1.1-cp311-cp311-win_amd64.whl


# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

# pip install --no-deps trl peft accelerate bitsandbytes

# 找到下载的trion: https://hf-mirror.com/madbuda/triton-windows-builds

# 然后
pip install triton-2.1.0-cp311-cp311-win_amd64.whl

# pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
pip install git+https://github.com/huggingface/trl.git

# 
pip install ipykernel 
pip install jupyter 
pip install jupyterlab 

"""

print('cuda:', torch.version.cuda, '\nPytorch:', torch.__version__)

# import torch; 

# print('Using device: ', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))