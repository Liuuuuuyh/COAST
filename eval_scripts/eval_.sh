# num_proce=4
# gpu_id=0,1,2,3

num_proce=8
gpu_id=0,1,2,3,4,5,6,7

export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token $HF_TOKEN
CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
    --main_process_port 29502 \
    --num_processes=$num_proce \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/llava-v1.5-7b" \
    --tasks mme \
    --batch_size 1 \
    --output_path ./logs/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks mmbench_en \
#     --batch_size 1 \
#     --output_path ./logs/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29507 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks ai2d \
#     --batch_size 1 \
#     --output_path ./logs/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks gqa \
#     --batch_size 1 \
#     --output_path ./logs/ 

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks vizwiz_vqa \
#     --batch_size 1 \
#     --output_path ./logs/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --output_path ./logs/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29507 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks scienceqa_img \
#     --batch_size 1 \
#     --output_path ./logs/











# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks gqa,scienceqa_img,pope \
#     --batch_size 1 \
#     --output_path ./logs/ 

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29507 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks ai2d \
#     --batch_size 1 \
#     --output_path ./logs/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks vizwiz_vqa \
#     --batch_size 1 \
#     --output_path ./logs/


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29507 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks ocrbench \
#     --batch_size 1 \
#     --output_path ./logs/


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks ai2d,infovqa \
#     --batch_size 1 \
#     --output_path ./logs/


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks ocrbench,chartqa \
#     --batch_size 1 \
#     --output_path ./logs/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks docvqa,realworldqa \
#     --batch_size 1 \
#     --output_path ./logs/ 

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29502 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks vizwiz_vqa \
#     --batch_size 1 \
#     --output_path ./logs/

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
#     --main_process_port 29507 \
#     --num_processes=$num_proce \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/path/to/llava-v1.5-7b" \
#     --tasks textvqa \
#     --batch_size 1 \
#     --output_path ./logs/