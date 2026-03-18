"""
Pre-cache HuggingFace models for offline use on Leonardo compute nodes.

Run on the login node (which has internet):
    export HF_HOME=$WORK/hf_cache
    python scripts/leonardo/download_hf_models.py

Downloads Florence-2-large and CLIP (used by LIBERO for task embeddings).
"""

import os
import sys


def main():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.environ.get("HOME", ""), ".cache", "huggingface"),
    )
    print(f"HuggingFace cache directory: {cache_dir}")

    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, CLIPModel

    # Florence-2
    model_name = "microsoft/Florence-2-large"
    print(f"\nDownloading {model_name}...")
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"  Processor cached: {type(processor).__name__}")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        print(f"  Model cached: {type(model).__name__}")
    except Exception as e:
        print(f"ERROR downloading {model_name}: {e}", file=sys.stderr)
        sys.exit(1)

    # CLIP (used by LIBERO for task embeddings)
    clip_name = "openai/clip-vit-base-patch32"
    print(f"\nDownloading {clip_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(clip_name)
        print(f"  Tokenizer cached: {type(tokenizer).__name__}")
        clip_model = CLIPModel.from_pretrained(clip_name)
        print(f"  Model cached: {type(clip_model).__name__}")
    except Exception as e:
        print(f"ERROR downloading {clip_name}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll models cached to {cache_dir}")
    print("Compute nodes can now load with TRANSFORMERS_OFFLINE=1")


if __name__ == "__main__":
    main()
