import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_DIRS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "models/llm/tinyllama-1.1b-chat",
    "openai-community/gpt2-large": "models/llm/gpt2-large",
    "facebook/opt-1.3b": "models/llm/opt-1.3b",
}


def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model for SMETimes.")
    parser.add_argument(
        "--repo_id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id, for example TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    parser.add_argument("--local_dir", default=None, help="Directory to store model files.")
    parser.add_argument("--revision", default=None, help="Optional model revision/branch/tag.")
    args = parser.parse_args()

    local_dir = args.local_dir or DEFAULT_MODEL_DIRS.get(
        args.repo_id,
        "models/llm/" + args.repo_id.replace("/", "__"),
    )
    local_dir = Path(local_dir).resolve()
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(local_dir),
    )
    print(f"Downloaded {args.repo_id} to {path}")


if __name__ == "__main__":
    main()
