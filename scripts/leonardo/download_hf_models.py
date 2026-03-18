"""
Pre-cache HuggingFace models for offline use on Leonardo compute nodes.

Run on the login node (which has internet):
    export HF_HOME=$WORK/hf_cache
    python scripts/leonardo/download_hf_models.py

Downloads Florence-2-large (model + processor) to the HF cache directory.
"""

import os
import sys


def main():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.environ.get("HOME", ""), ".cache", "huggingface"),
    )
    print(f"HuggingFace cache directory: {cache_dir}")

    model_name = "microsoft/Florence-2-large"
    print(f"\nDownloading {model_name}...")

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        print(f"  Processor cached: {type(processor).__name__}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        print(f"  Model cached: {type(model).__name__}")

    except Exception as e:
        print(f"ERROR downloading {model_name}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll models cached to {cache_dir}")
    print("Compute nodes can now load with TRANSFORMERS_OFFLINE=1")


if __name__ == "__main__":
    main()
