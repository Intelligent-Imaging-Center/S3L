# S3L

<h1 align="center"> Leveraging Spatiotemporal Cues for Self-Supervised Stereo Depth Estimation in Endoscopic Videos </h1>

We have released the segmentation configurations and loading scripts for the SCARED dataset in advance.

## 📝 Checklist

- ⬛ Upload data processing code
- ⬛ Upload inference code 
- ⬛ Upload Upload pre-trained weights  
- ⬛ Upload training code  

## 🛠 Installation
1. First you have to make sure that you clone the repo with the `--recursive` flag.
```bash
git clone --recursive https://github.com/Intelligent-Imaging-Center/S3L.git
cd S3L
```
2. Creating a new conda environment.
```bash
conda create --name s3l python=3.9
conda activate s3l
```
3. Install CUDA 11.8 and torch-related pacakges
```bash
pip install numpy==1.25.0 # do not use numpy >= v2.0.0
conda install --channel "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```
4. Now install the other requirements
```bash
conda env update -f environment.yml --prune
```
### Basic Dependencies:
- CUDA Version >= 11.8
- Python >= 3.8
- Pytorch >= 2.0.0

## 📁 Dataset Preparation
We excluded subsets D4 and D5 of the SCARED dataset due to severe calibration inaccuracies and temporal misalignment. The remaining subsets were split as follows: D1–D3 and D6–D7 were used for training, yielding 19 videos with a total of 17,206 frames. D8 and D9 were reserved for testing, providing 8 videos with 5,907 frames.

Please refer to [this](https://github.com/EikoLoki/MICCAI_challenge_preprocess) to prepare your SCARED data.
```bash
The folder structure is as follows:
scard/
├── dataset_1/
│   ├── keyframe_1/
│   │   ├── disp/
│   │   ├── left_finalpass/
│   │   └── right_finalpass/
│   ├── keyframe_2/
│   ├── keyframe_3/
│   └── keyframe_4/
├── dataset_2/
│   ├── keyframe_1/
│   ├── keyframe_2/
│   └── keyframe_3/
...
└── dataset_9/
    ├── keyframe_1/
    ├── keyframe_2/
    └── keyframe_3/
```
## ✨ Quick Test:
preparing trained model：We will release the pre-trained model soon.
start testing single Image
```bash
python eval_img.py --load_weights path/to/your/weights/folder --image_path path/to/your/test/image
```
start testing vedio
```bash
python eval_vedio.py --load_weights path/to/your/weights/folder --vedio_path path/to/your/test/vedio
```
## 🖋 Train:
```bash
python train.py --data_path path/to/your/data --output_name mytrain --config configs/scared/d1/k1.yaml
```
