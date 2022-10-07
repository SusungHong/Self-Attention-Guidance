SAMPLE_FLAGS="--batch_size 16 --num_samples 10000 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAG_FLAGS="--guide_scale 1.5 --guide_start 250 --sel_attn_block output --sel_attn_depth 2 --blur_sigma 9 --classifier_guidance False"
mpiexec -n 8 python classifier_sample.py $SAG_FLAGS $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS
