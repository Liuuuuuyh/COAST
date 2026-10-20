num_proce=4
gpu_id=4,5,6,7

CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
    --main_process_port 29502 \
    --num_processes=$num_proce \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/llava-v1.5-7b" \
    --tasks mme \
    --batch_size 1 \
    --output_path ./logs/base_llava/ 


CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
    --main_process_port 29502 \
    --num_processes=$num_proce \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/llava-v1.5-7b" \
    --tasks mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme  \
    --output_path ./logs/base_llava/ 


CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
    --main_process_port 29502 \
    --num_processes=$num_proce \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/llava-v1.5-7b" \
    --tasks gqa,scienceqa_img,pope \
    --batch_size 1 \
    --output_path ./logs/base_llava/ 


CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
    --main_process_port 29502 \
    --num_processes=$num_proce \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/llava-v1.5-7b" \
    --tasks ai2d,chartqa \
    --batch_size 1 \
    --output_path ./logs/base_llava/


CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
    --main_process_port 29502 \
    --num_processes=$num_proce \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/llava-v1.5-7b" \
    --tasks ocrbench,infovqa \
    --batch_size 1 \
    --output_path ./logs/base_llava/


CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
    --main_process_port 29502 \
    --num_processes=$num_proce \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/llava-v1.5-7b" \
    --tasks docvqa,realworldqa \
    --batch_size 1 \
    --output_path ./logs/base_llava/ 


CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
    --main_process_port 29502 \
    --num_processes=$num_proce \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/llava-v1.5-7b" \
    --tasks vizwiz_vqa \
    --batch_size 1 \
    --output_path ./logs/base_llava/