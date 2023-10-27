# Self-Attention Diffusion Guidance (ICCV`23)
<a href="https://arxiv.org/abs/2210.00939"><img src="https://img.shields.io/badge/arXiv-2210.00939-%23B31B1B"></a>
<a href="https://ku-cvlab.github.io/Self-Attention-Guidance"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
<a href="https://huggingface.co/spaces/susunghong/Self-Attention-Guidance"><img src="https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565"></a>
<!-- <a href="https://colab.research.google.com/github/SusungHong/Self-Attention-Guidance/blob/main/SAG_Stable.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a> -->

![image](https://user-images.githubusercontent.com/5498512/203083063-b61df338-c986-4980-81f0-1f1532ea8245.png)
This is the implementation of the paper <a href="https://arxiv.org/abs/2210.00939">Improving Sample Quality of Diffusion Models Using Self-Attention Guidance</a> by Hong et al. To gain insight from our exploration of the self-attention maps of diffusion models and for detailed explanations, please see our [Paper](https://arxiv.org/abs/2210.00939) and [Project Page](https://ku-cvlab.github.io/Self-Attention-Guidance).

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), and we modified feature extraction code from [yandex-research/ddpm-segmentation](https://github.com/yandex-research/ddpm-segmentation) to get the self-attention maps. The major implementation of our method is in `./guided_diffusion/gaussian_diffusion.py` and `./guided_diffusion/unet.py`.

All you need is to setup the environment, download existing models, and sample from them using our implementation. Neither further training nor a dataset is needed to apply self-attention guidance!

## Updates

**2023-08-14:** This repository supports DDIM sampling with SAG.

**2023-02-19:** The [Gradio Demo](https://huggingface.co/spaces/susunghong/Self-Attention-Guidance):hugs: of SAG for Stable Diffusion is now available

**2023-02-16:** The Stable Diffusion pipeline of SAG is now available at [huggingface/diffusers](https://huggingface.co/docs/diffusers/api/pipelines/self_attention_guidance) :hugs::firecracker:

**2023-02-01:** The demo for Stable Diffusion is now available in [Colab](https://colab.research.google.com/github/SusungHong/Self-Attention-Guidance/blob/main/SAG_Stable.ipynb).

## Environment
* Python 3.8, PyTorch 1.11.0
* 8 x NVIDIA RTX 3090 (set `backend="gloo"` in `./guided_diffusion/dist_util.py` if P2P access is not available)
```
git clone https://github.com/KU-CVLAB/Self-Attention-Guidance
conda create -n sag python=3.8 anaconda
conda activate sag
conda install mpi4py
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install blobfile
```

## Downloading Pretrained Diffusion Models (and Classifiers for CG)
Pretrained weights for ImageNet and LSUN can be downloaded from [the repository](https://github.com/openai/guided-diffusion). Download and place them in the `./models/` directory.

## Sampling from Pretrained Diffusion Models
You can sample from pretrained diffusion models with self-attention guidance by changing `SAG_FLAGS` in the following commands. Note that sampling with `--guide_scale 1.0` means sampling without self-attention guidance. Below are the 4 examples.

 * ImageNet 128x128 model (`--classifier_guidance False` deactivates classifier guidance):
```
SAMPLE_FLAGS="--batch_size 64 --num_samples 10000 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAG_FLAGS="--guide_scale 1.1 --guide_start 250 --sel_attn_block output --sel_attn_depth 8 --blur_sigma 3 --classifier_guidance True"
mpiexec -n $NUM_GPUS python classifier_sample.py $SAG_FLAGS $MODEL_FLAGS --classifier_scale 0.5 --classifier_path models/128x128_classifier.pt --model_path models/128x128_diffusion.pt $SAMPLE_FLAGS
```

 * ImageNet 256x256 model (`--class_cond True` for conditional models):
```
SAMPLE_FLAGS="--batch_size 16 --num_samples 10000 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAG_FLAGS="--guide_scale 1.5 --guide_start 250 --sel_attn_block output --sel_attn_depth 2 --blur_sigma 9 --classifier_guidance False"
mpiexec -n $NUM_GPUS python classifier_sample.py $SAG_FLAGS $MODEL_FLAGS --classifier_scale 0.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS
```

 * LSUN Cat model (respaced to 250 steps):
```
SAMPLE_FLAGS="--batch_size 16 --num_samples 10000 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAG_FLAGS="--guide_scale 1.05 --guide_start 250 --sel_attn_block output --sel_attn_depth 2 --blur_sigma 9 --classifier_guidance False"
mpiexec -n $NUM_GPUS python image_sample.py $SAG_FLAGS $MODEL_FLAGS --model_path models/lsun_cat.pt $SAMPLE_FLAGS
```

 * LSUN Horse model (respaced to 250 steps):
```
SAMPLE_FLAGS="--batch_size 16 --num_samples 10000 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAG_FLAGS="--guide_scale 1.01 --guide_start 250 --sel_attn_block output --sel_attn_depth 2 --blur_sigma 9 --classifier_guidance False"
mpiexec -n $NUM_GPUS python image_sample.py $SAG_FLAGS $MODEL_FLAGS --model_path models/lsun_horse.pt $SAMPLE_FLAGS
```

 * ImageNet 128x128 model (DDIM 25 steps):
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 128 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 8 --num_samples 8 --timestep_respacing ddim25 --use_ddim True"
SAG_FLAGS="--guide_scale 1.1 --guide_start 25 --sel_attn_block output --sel_attn_depth 8 --blur_sigma 3 --classifier_guidance True"
mpiexec -n $NUM_GPUS python classifier_sample.py \
    --model_path models/128x128_diffusion.pt \
    --classifier_path models/128x128_classifier.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS $SAG_FLAGS
```

# Results

**Compatibility of self-attention guidance (SAG) and classifier guidance (CG) on ImageNet 128x128 model:**

| SAG | CG | FID | sFID | Precision | Recall |
|---|---|---|---|---|---|
|  |  | 5.91 | 5.09 | 0.70 | 0.65 |
|  | V | 2.97 | 5.09 | 0.78 | 0.59 |
| V |  | 5.11 | 4.09 | 0.72 | 0.65 |
| V | V | 2.58 | 4.35 | 0.79 | 0.59 |

**Results on pretrained models:**

| Model | # of steps | Self-attention guidance scale | FID | sFID | IS | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ImageNet 256×256 (Uncond.) | 250 | 0.0 (baseline)<br>0.5<br>0.8 | 26.21<br>20.31<br>20.08 | 6.35<br>5.09<br>5.77 | 39.70<br>45.30<br>45.56 | 0.61<br>0.66<br>0.68 | 0.63<br>0.61<br>0.59 |
| ImageNet 256×256 (Cond.) | 250 | 0.0 (baseline)<br>0.2 | 10.94<br>9.41 | 6.02<br>5.28 | 100.98<br>104.79 | 0.69<br>0.70 | 0.63<br>0.62 |
| LSUN Cat 256×256 | 250 | 0.0 (baseline)<br>0.05 | 7.03<br>6.87 | 8.24<br>8.21 | -<br>- | 0.60<br>0.60 | 0.53<br>0.50 |
| LSUN Horse 256×256 | 250 | 0.0 (baseline)<br>0.01 | 3.45<br>3.43 | 7.55<br>7.51 | -<br>- | 0.68<br>0.68 | 0.56<br>0.55 |

# Cite as
```
@inproceedings{hong2023improving,
  title={Improving sample quality of diffusion models using self-attention guidance},
  author={Hong, Susung and Lee, Gyuseong and Jang, Wooseok and Kim, Seungryong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7462--7471},
  year={2023}
}
```
