# Environment notes and known fixes

## FP8 / GPTNeoX import error

If you see an error like:

```text
AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'
ModuleNotFoundError: Could not import module 'GPTNeoXForCausalLM'
```

this is almost certainly a version mismatch between `transformers` and `torch`, not a Pythia or GPT-NeoX modelling issue.

For this project we do not need the newest Transformers FP8 integrations. Use the pinned E1-safe stack:

```bash
pip uninstall -y transformers accelerate huggingface_hub tokenizers safetensors
pip install -r requirements.txt
pip install -e .
python scripts/check_environment.py
```

In notebook/Colab environments, restart the runtime/kernel after reinstalling packages.

## Recommended CUDA install, if torch itself needs reinstalling

For CUDA 12.1 wheels:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 'torch>=2.1,<2.8'
```

Then install the rest:

```bash
pip install -r requirements.txt
pip install -e .
```
