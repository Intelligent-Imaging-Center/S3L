# S3L

<h1 align="center"> Leveraging Spatiotemporal Cues for Self-Supervised Stereo Depth Estimation in Endoscopic Videos </h1>

We have released the segmentation configurations and loading scripts for the SCARED dataset in advance.

## 📝 Checklist

- ⬛ Upload data processing code
- ⬛ Upload inference code 
- ⬛ Upload Upload pre-trained weights  
- ⬛ Upload training code  

## 🛠 Installation
### Basic Dependencies:
- CUDA Version >= 11.8
- Python >= 3.8
- Pytorch >= 2.0.0

### Create a new environment:

```bash
conda create -n s3l python==3.9
conda activate s3l
```
## 📁 Dataset Preparation

We excluded subsets D4 and D5 of the SCARED dataset due to severe calibration inaccuracies and temporal misalignment. The remaining subsets were split as follows: D1–D3 and D6–D7 were used for training, yielding 19 videos with a total of 17,206 frames. D8 and D9 were reserved for testing, providing 8 videos with 5,907 frames.
