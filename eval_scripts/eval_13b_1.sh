num_proce=1
gpu_id=6,7

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-13b,device_map=auto" \
#     --tasks mme \
#     --batch_size 1 \
#     --output_path ./logs/fastv/ 

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-13b,device_map=auto" \
#     --tasks mmbench_en \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.5_mme  \
#     --output_path ./logs/fastv/ 

export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token $HF_TOKEN
CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
    --main_process_port 29504 \
    --num_processes=$num_proce \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/llava-v1.5-13b,device_map=auto" \
    --tasks gqa,scienceqa_img,pope \
    --batch_size 1 \
    --output_path ./logs/fastv/ 

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-13b,device_map=auto" \
#     --tasks ai2d,chartqa \
#     --batch_size 1 \
#     --output_path ./logs/fastv/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29503 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-13b,device_map=auto" \
#     --tasks ocrbench,infovqa \
#     --batch_size 1 \
#     --output_path ./logs/fastv/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29503 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-13b,device_map=auto" \
#     --tasks docvqa,realworldqa \
#     --batch_size 1 \
#     --output_path ./logs/fastv/ 

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29503 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-13b,device_map=auto" \
#     --tasks vizwiz_vqa \
#     --batch_size 1 \
#     --output_path ./logs/fastv/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29503 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-13b,device_map=auto" \
#     --tasks textvqa \
#     --batch_size 1 \
#     --output_path ./logs/fastv/