import os

import requests


def download_fasttext(self):
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    # destination folder can be configured with the FASTTEXT_DIR environment variable
    folder = os.environ.get("FASTTEXT_DIR", os.path.join("models", "fasttext"))
    os.makedirs(folder, exist_ok=True)

    filename = os.path.basename(url)
    dest_path = os.path.join(folder, filename)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dest_path
