python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/llava-v1.5-7b" \
    --tasks mme \
    --batch_size 1 \
    --output_path ./logs/
