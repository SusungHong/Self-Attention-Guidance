# Self-Attention Diffusion Guidance
![image](https://user-images.githubusercontent.com/5498512/194516853-18048309-06c1-4e7c-9a3d-9d911186272a.png)

This is the implementation of the paper <a href="https://arxiv.org/abs/2210.00939">Improving Sample Quality of Diffusion Models Using Self-Attention Guidance</a> by Susung Hong, Gyuseong Lee, Wooseok Jang and Seungryong Kim. To gain insight from our exploration of the self-attention maps of diffusion models and for further details, please see our [Paper](https://arxiv.org/abs/2210.00939) and [Project Page](https://ku-cvlab.github.io/Self-Attention-Guidance).

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), and you need to additionally install `torchvision` package on that environmental settings. The major implementation of our method is in `guided_diffusion/gaussian_diffusion.py` and `guided_diffusion/unet.py`. We modified feature extraction code from [yandex-research/ddpm-segmentation](https://github.com/yandex-research/ddpm-segmentation) to get the self-attention maps.

All you need is to download pretrained models, and sample from them using our implementation. Neither a trainable module nor a dataset is needed to use self-attention guidance!

# Downloading Pretrained Diffusion Models (and Classifiers for CG)
Pretrained weights for ImageNet and LSUN can be downloaded from [the repository](https://github.com/openai/improved-diffusion). Download and place them in the `./models/` directory.

# Sampling from Pretrained Diffusion Models
You can sample from pretrained diffusion models with self-attention guidance by changing `SAG_FLAGS` in the following commands. Note that sampling with `--guide_scale 1.0` means sampling without self-attention guidance.

 * ImageNet 128x128 model (`--classifier_guidance False` deactivates classifier guidance):
```
SAMPLE_FLAGS="--batch_size 64 --num_samples 10000 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAG_FLAGS="--guide_scale 1.1 --guide_start 250 --sel_attn_block output --sel_attn_depth 8 --blur_sigma 3 --classifier_guidance True"
mpiexec -n 8 python classifier_sample.py $SAG_FLAGS $MODEL_FLAGS --classifier_scale 0.5 --classifier_path models/128x128_classifier.pt --model_path models/128x128_diffusion.pt $SAMPLE_FLAGS
```

 * ImageNet 256x256 model (`--class_cond True` for conditional models):
```
SAMPLE_FLAGS="--batch_size 16 --num_samples 10000 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAG_FLAGS="--guide_scale 1.5 --guide_start 250 --sel_attn_block output --sel_attn_depth 2 --blur_sigma 9 --classifier_guidance False"
mpiexec -n 8 python classifier_sample.py $SAG_FLAGS $MODEL_FLAGS --classifier_scale 0.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS
```

 * LSUN Cat model (respaced to 250 steps):
```
SAMPLE_FLAGS="--batch_size 16 --num_samples 10000 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAG_FLAGS="--guide_scale 1.05 --guide_start 250 --sel_attn_block output --sel_attn_depth 2 --blur_sigma 9 --classifier_guidance False"
mpiexec -n 8 python image_sample.py $SAG_FLAGS $MODEL_FLAGS --model_path models/lsun_cat.pt $SAMPLE_FLAGS
```

 * LSUN Horse model (respaced to 250 steps):
```
SAMPLE_FLAGS="--batch_size 16 --num_samples 10000 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAG_FLAGS="--guide_scale 1.01 --guide_start 250 --sel_attn_block output --sel_attn_depth 2 --blur_sigma 9 --classifier_guidance False"
mpiexec -n 8 python image_sample.py $SAG_FLAGS $MODEL_FLAGS --model_path models/lsun_horse.pt $SAMPLE_FLAGS
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
@article{hong2022improving,
  title={Improving Sample Quality of Diffusion Models Using Self-Attention Guidance},
  author={Hong, Susung and Lee, Gyuseong and Jang, Wooseok and Kim, Seungryong},
  journal={arXiv preprint arXiv:2210.00939},
  year={2022}
}
```
