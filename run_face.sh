ckpt="download/stylegan2-ffhq-config-f.pt"
prompt="3d human face, closeup cute and adorable, cute big circular reflective eyes, Pixar render, unreal engine cinematic smooth, intricate detail, cinematic"
img_size=1024


'''
StyleGAN-Fusion (no regularizer)
'''
exp_name="face_prompt0_Diffusion_minStep10_maxStep500_NoReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 1001  \
--target_class "${prompt}"  \
--output_interval 50  \
--save_interval 10000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--num_inference_steps 50 \
--Disable_lpip \


'''
StyleGAN-Fusion (directional regularizer)
'''
exp_name="face_prompt0_Diffusion_minStep10_maxStep500_DirectionalReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 1001  \
--target_class "${prompt}"  \
--output_interval 50  \
--save_interval 10000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--num_inference_steps 50 \
--Disable_lpip \
--Directional_loss 1 \


'''
StyleGAN-Fusion (reconstruction regularizer)
'''
exp_name="face_prompt0_Diffusion_minStep10_maxStep500_ReconstructionReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 1001  \
--target_class "${prompt}"  \
--output_interval 50  \
--save_interval 10000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--num_inference_steps 50 \
--Disable_lpip \
--Directional_loss 2 \


'''
StyleGAN-Fusion (LPIPS regularizer)
'''
exp_name="face_prompt0_Diffusion_minStep10_maxStep500_LpipsReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 1001  \
--target_class "${prompt}"  \
--output_interval 50  \
--save_interval 10000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--num_inference_steps 50 \


######################################
######   Layer selections
######################################
'''
StyleGAN-Fusion (select 12 layers)
'''
exp_name="face_prompt0_DiffusionLayerSelection_L12_minStep10_maxStep500_NoReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 1201  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 12 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 50  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 0 \

exp_name="face_prompt0_DiffusionLayerSelection_L12_minStep10_maxStep500_DirectionalReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 1201  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 12 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 50  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 1 \

exp_name="face_prompt0_DiffusionLayerSelection_L12_minStep10_maxStep500_ReconstructionReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 1201  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 12 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 50  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 2 \


'''
StyleGAN-Fusion (select 6 layers)
'''
exp_name="face_prompt0_DiffusionLayerSelection_L6_minStep10_maxStep500_NoReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 3001  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 6 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 50  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 0 \

exp_name="face_prompt0_DiffusionLayerSelection_L6_minStep10_maxStep500_DirectionalReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 3001  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 6 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 50  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 1 \

exp_name="face_prompt0_DiffusionLayerSelection_L6_minStep10_maxStep500_ReconstructionReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 3001  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 6 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 50  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 2 \


'''
StyleGAN-Fusion (select 3 layers)
'''
exp_name="face_prompt0_DiffusionLayerSelection_L3_minStep10_maxStep500_NoReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 5001  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 3 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 50  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 0 \

exp_name="face_prompt0_DiffusionLayerSelection_L3_minStep10_maxStep500_DirectionalReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 5001  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 3 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 50  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 1 \

exp_name="face_prompt0_DiffusionLayerSelection_L3_minStep10_maxStep500_ReconstructionReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 5001  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 3 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 50  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 2 \


'''
StyleGAN-Fusion (select 1 layer)
'''
exp_name="face_prompt0_DiffusionLayerSelection_L1_minStep10_maxStep500_NoReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 10001  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 1 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 100  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 0 \

exp_name="face_prompt0_DiffusionLayerSelection_L1_minStep10_maxStep500_DirectionalReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 10001  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 1 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 100  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 1 \

exp_name="face_prompt0_DiffusionLayerSelection_L1_minStep10_maxStep500_ReconstructionReg"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 10001  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 1 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 100  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 1000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 1.0 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Disable_lpip \
--Layer_selection_Diffusion \
--Directional_loss 2 \


'''
StyleGAN-NADA
'''
exp_name="face_prompt0_NaDa"
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 2001  \
--source_class "photo"  \
--target_class "${prompt}"  \
--auto_layer_k 18 \
--auto_layer_iters 1  \
--auto_layer_batch 1  \
--output_interval 50  \
--clip_models "ViT-B/32" "ViT-B/16"  \
--clip_model_weights 1.0 1.0  \
--mixing 0.0 \
--save_interval 10000000 \


'''
StyleGAN-Fusion (EG3D)
'''
exp_name="face_0_3D_Diffusion_minStep10_maxStep500_DirectionalReg_w05_LpipReg"
ckpt='download/eg3d/ffhq512-128.pkl'
CUDA_VISIBLE_DEVICES=0 python train.py --size ${img_size} \
--batch 1 \
--n_sample 4 \
--output_dir logs/${exp_name} \
--lr 0.002  \
--frozen_gen_ckpt ${ckpt}  \
--iter 5001  \
--target_class "${prompt}"  \
--output_interval 100  \
--save_interval 10000000 \
--diffusion \
--freeze_bias \
--CFG 100 \
--min_step 10 \
--max_step 500 \
--directional_loss_weight_multiplicity 0.5 \
--lpip_weight_multiplicity 1.0 \
--num_inference_steps 50 \
--rescale_CFGDirectionNoise \
--Directional_loss 1 \
--rescaleCFG_Noise \
--Disable_ImageGradClamp \
--EG3D 
