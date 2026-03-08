<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## Tesla P40 Build And Run

This fork is patched to run vLLM on NVIDIA Tesla P40 / Pascal (`sm_61`) and has been validated with `Qwen/Qwen3-ASR-1.7B`.

Tested environment:

- GPU: Tesla P40 (`sm_61`)
- Python: `3.12`
- PyTorch: `2.5.1+cu121`
- CUDA toolkit used for build: `12.1`
- Recommended install mode: in a dedicated `venv`, then `pip install -e . --no-build-isolation`

The commands below are the reproducible path used to build and run this fork on P40.

### 1. Clone the fork

```bash
git clone https://github.com/uaysk/vllm-pascal.git
cd vllm-pascal
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv-p40
source .venv-p40/bin/activate
python -m pip install --upgrade pip
```

### 3. Install build tools

```bash
python -m pip install \
  "setuptools>=77,<81" \
  wheel \
  packaging \
  cmake \
  ninja \
  jinja2 \
  regex \
  protobuf
```

### 4. Install a P40-compatible PyTorch stack

The fork has been tested with CUDA 12.1 wheels:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1
```

Verify that PyTorch sees the P40 correctly:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

Expected capability on Tesla P40:

```text
(6, 1)
```

### 5. Build and install this fork for Pascal

```bash
export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST="6.1"
python -m pip install -e . --no-build-isolation
```

If you want to reduce parallel build pressure on an older host:

```bash
export MAX_JOBS=1
```

and then run the same install command again.

### 6. Install audio runtime packages used for Qwen ASR validation

```bash
python -m pip install librosa soundfile
```

### 7. Start the OpenAI-compatible API server for Qwen3-ASR on P40

For P40, the stable runtime path is eager execution with conservative scheduler settings:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve Qwen/Qwen3-ASR-1.7B \
  --host 0.0.0.0 \
  --port 8010 \
  --served-model-name qwen3-asr-rt \
  --hf-overrides '{"architectures":["Qwen3ASRRealtimeGeneration"]}' \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"audio":1}' \
  --enforce-eager \
  --disable-log-stats \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --no-async-scheduling
```

If you already have the model in a local Hugging Face snapshot directory, replace `Qwen/Qwen3-ASR-1.7B` with that local path.

### 8. Verify basic health

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

### 9. Known P40-specific runtime notes

- This fork intentionally avoids FlashAttention/FlashInfer-style fast paths that are not usable on Pascal.
- For Qwen3-ASR on P40, use the serve flags shown above. Removing them can reintroduce crashes or unstable behavior.
- The validated path is vLLM serving plus Qwen3-ASR transcription / realtime transcription on P40.
- Browser microphone capture still requires `localhost` or `HTTPS`; that is a browser security rule, not a vLLM limitation.

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
