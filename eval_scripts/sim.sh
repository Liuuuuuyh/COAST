gpu_id=0,1

# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
#                                 --dataset_name mme\
#                                 --version_ 13


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
#                                 --dataset_name mmbench_en\
#                                 --version_ 13


export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token $HF_TOKEN
CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
                                --dataset_name gqa\
                                --version_ 13


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
#                                 --dataset_name scienceqa_img\
#                                 --version_ 13


export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token $HF_TOKEN
CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
                                --dataset_name pope\
                                --version_ 13


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
#                                 --dataset_name ai2d\
#                                 --version_ 13


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
#                                 --dataset_name chartqa\
#                                 --version_ 13


# # export HF_ENDPOINT=https://hf-mirror.com
# # hf auth login --token $HF_TOKEN
# # CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
# #                                 --dataset_name ocrbench\
# #                                 --version_ 13


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
#                                 --dataset_name infovqa\
#                                 --version_ 13


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
#                                 --dataset_name docvqa\
#                                 --version_ 13


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
#                                 --dataset_name realworldqa\
#                                 --version_ 13


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
#                                 --dataset_name vizwiz_vqa\
#                                 --version_ 13


# export HF_ENDPOINT=https://hf-mirror.com
# hf auth login --token $HF_TOKEN
# CUDA_VISIBLE_DEVICES=$gpu_id python analy_datasets.py \
#                                 --dataset_name textvqa\
#                                 --version_ 13



