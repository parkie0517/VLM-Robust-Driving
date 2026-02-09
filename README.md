# VLM-Robust-Driving
Leveraging Large Vision-Language Models for Robust Autonomous Driving under Adverse Weather and Anomalies

## Authors
This project is developed by:
- **Sanghyeon Kim** (Division of Future Vehicle)  
- **Heejun Park** (Division of Future Vehicle)  
- **Jinsu Ra** (Division of Future Vehicle)  
- **Yubin Lee** (CCS Graduate School of Mobility)  

## Download pretrained weight
```
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

## Install
```
# conda
conda remove -n hj_dino --all -y
conda create -n hj_dino python=3.9 -y
conda activate hj_dino

# cuda
conda install nvidia/label/cuda-11.8.0::cuda -y
conda install cudatoolkit=11.8.0 -c nvidia -y
export CUDA_HOME=/home/user/anaconda3/envs/hj_dino
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo $CUDA_HOME # /home/user/anaconda3/envs/hj_dino 나와야 함
nvcc -V # cuda 11.8 나와야 함

# pytorch
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y


# etc
pip install pip==22.3.1
pip install "setuptools>=62.3.0,<75.9"
pip install numpy==1.26.0

# grounding dino
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/

pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

```

## How to run
```
conda activate hj_dino

python test.py # detection result

python main_infer_v4.py # run to infer for the whole dataset
```

## Acknowledgement
Many thanks to these excellent open source projects:

-  [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)