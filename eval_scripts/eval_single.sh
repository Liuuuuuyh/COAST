
#### 7 b
num_proce=8
gpu_id=0,1,2,3,4,5,6,7
# num_proce=4
# gpu_id=4,5,6,7
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
    --output_path ./logs/base_llava/


### 13 b
# num_proce=1
# gpu_id=0,1,2,3,4,5,6,7
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
#     --output_path ./logs/base_llava/ 