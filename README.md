# COAST: Contrastive Adaptive Semantic Token Pruning

Official implementation of **Evading Visual Aphasia: Contrastive Adaptive Semantic Token Pruning for Vision-Language Models**. COAST is a training-free visual-token pruning method for LVLM inference. It preserves query-relevant anchors, estimates contextual dispersion with attention entropy, and routes remaining tokens with a contrastive semantic/context score.

This release is based on LLaVA 1.5. Checkpoints and benchmark datasets are not included; obtain them from their original providers and pass local paths to the commands below.

## Method

At configured decoder layers, COAST (1) extracts high-attention semantic anchors and low-relevance negative anchors from captured text-to-image attention, (2) adapts the semantic/context retention budget using normalized attention entropy, and (3) scores candidate tokens by `MaxSim(positive anchors) - MeanSim(negative anchors)`. High-scoring candidates provide semantic evidence and low-scoring candidates provide complementary context. Pruning progressively shortens the sequence.

The default LLaVA-1.5 336px setup uses image token span `[35, 611)` and keeps 214, 78, 6, and 4 image tokens after decoder layers 2, 12, 22, and 28. These positions depend on the vision encoder, image resolution, prompt layout, and multimodal token insertion; verify them if changing the setup.

## Installation

The dependency pins in `pyproject.toml` follow the upstream LLaVA 1.5 setup (including PyTorch 2.1.2 and Transformers 4.37.2).

```bash
conda create -n coast-llava python=3.10 -y
conda activate coast-llava
# Install the PyTorch wheel matching your CUDA version first, if needed.
git clone <THIS_REPOSITORY_URL> COAST_LLaVA
cd COAST_LLaVA
pip install --upgrade pip
pip install -e .
```

## Single-image inference

```bash
python -m llava.eval.run_llava \
  --model-path /path/to/llava-v1.5-7b \
  --image-file /path/to/image.jpg \
  --query "Describe the image." \
  --conv-mode llava_v1
```

The LLaVA Llama model loader imports `COASTLlamaModel`, so configured COAST pruning stages are applied during generation. This command uses deterministic decoding with the default temperature.

## Benchmark evaluation

Install and configure `lmms-eval` and its benchmark datasets first. Example for MME:

```bash
CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  --model llava \
  --model_args pretrained=/path/to/llava-v1.5-7b \
  --tasks mme \
  --batch_size 1 \
  --output_path ./logs/coast/mme
```

For distributed evaluation, use `accelerate launch --num_processes <N> -m lmms_eval ...` and set `CUDA_VISIBLE_DEVICES` to the selected devices. Task names and dataset setup depend on the installed `lmms-eval` version. Additional scripts are in `eval_scripts/`.

## Citation

```bibtex
@article{ma2026evading,
  title   = {Evading Visual Aphasia: Contrastive Adaptive Semantic Token Pruning for Vision-Language Models},
  author  = {Ma, Jie and Liu, Yihang and Qiu, Zhike and Ji, Jiayi and Sun, Xiaoshuai},
  journal = {arXiv preprint arXiv:2605.09429},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.09429},
  url     = {https://arxiv.org/abs/2605.09429}
}
```

Please also cite the [LLaVA paper](https://arxiv.org/abs/2304.08485) when using this code. The project includes upstream LLaVA components under the Apache License 2.0; see `LICENSE` and individual source notices.
