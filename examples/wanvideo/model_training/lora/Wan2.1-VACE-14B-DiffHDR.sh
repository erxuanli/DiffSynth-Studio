#!/bin/sh

DIFFSYNTH_SKIP_DOWNLOAD=false
DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.1-VACE-14B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.1-VACE-14B/metadata.csv \
  --data_file_keys "video,vace_video,vace_video_mask,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-14B-DiffHDR_lora" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2,before_proj,after_proj" \
  --lora_rank 32 \
  --extra_inputs "vace_video,vace_video_mask,vace_reference_image" \
  --use_gradient_checkpointing_offload
