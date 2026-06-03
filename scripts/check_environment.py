from __future__ import annotations

import sys


def main() -> None:
    print(f"python: {sys.version.split()[0]}")

    import torch
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    print(f"has torch.float8_e8m0fnu: {hasattr(torch, 'float8_e8m0fnu')}")

    import transformers
    print(f"transformers: {transformers.__version__}")

    from transformers import AutoModelForCausalLM
    print("AutoModelForCausalLM import: OK")

    # This is the architecture class used by Pythia.
    from transformers import GPTNeoXForCausalLM
    print("GPTNeoXForCausalLM import: OK")


if __name__ == "__main__":
    main()
