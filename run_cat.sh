gpu_id="0"
ckpt="download/afhqcat_rosinality.pt"
img_size=512

prompt='photo of a dog'

exp_name="cat_2dog_Diffusion_minStep10_maxStep300_NoReg"
CUDA_VISIBLE_DEVICES=1 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 2001  \
--target_class "${prompt}"  \
--output_interval 50  \
--save_interval 10000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 300 \
--num_inference_steps 50 \
--Disable_lpip \

exp_name="cat_2dog_Diffusion_minStep10_maxStep300_LpipsReg_w03"
CUDA_VISIBLE_DEVICES=1 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 2001  \
--target_class "${prompt}"  \
--output_interval 50  \
--save_interval 10000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 300 \
--num_inference_steps 50 \
--lpip_weight_multiplicity 0.3

exp_name="cat_2dog_Diffusion_minStep10_maxStep300_ReconstructionReg_w05"
CUDA_VISIBLE_DEVICES=1 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 2001  \
--target_class "${prompt}"  \
--output_interval 50  \
--save_interval 10000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 300 \
--num_inference_steps 50 \
--Disable_lpip \
--Directional_loss 2 \
--directional_loss_weight_multiplicity 0.5
