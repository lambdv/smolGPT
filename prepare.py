from pathlib import Path

import requests

DATASET_NAME = "shakespeare"
DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def prepare(dataset_name=DATASET_NAME):
    repo_root = Path(__file__).parent
    source_dir = repo_root / "nanoGPT" / "data" / dataset_name
    source_path = source_dir / "input.txt"
    target_dir = repo_root / "data"
    target_path = target_dir / "train.txt"

    if not source_path.exists():
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path.write_text(requests.get(DATASET_URL, timeout=30).text, encoding="utf-8")

    text = source_path.read_text(encoding="utf-8")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(text, encoding="utf-8")
    print(f"Prepared {dataset_name} dataset at {target_path}")
    return target_path


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else DATASET_NAME
    prepare(name)
