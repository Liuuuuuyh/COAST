export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token $HF_TOKEN
python arr_lmms.py run \
    --model-path /path/to/llava-v1.5-7b \
    --tasks gqa,pope,mme,mmbench_en,ai2d,scienceqa_img,ocrbench \
    --n-samples 10 \
    --output-dir ./arr_results


# 只画图（不需要 GPU）
# python arr_lmms.py plot \\
#     --results-json ./arr_results/arr_results.json \\
#     --output-dir ./arr_results