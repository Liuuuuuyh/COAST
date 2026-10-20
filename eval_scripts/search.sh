num_proce=8
gpu_id=0,1,2,3,4,5,6,7


export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token $HF_TOKEN
python search_params.py --target_avg_tokens 64 \
                        --n_trials 20 \
                        --gpus $gpu_id \
                        --num_processes $num_proce \
                        --port 29507