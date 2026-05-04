
import os
import gzip
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
 
import datasets as hf_datasets
 
from configure import (
    DATA_DIR, WIKI_DIR, INDEX_DIR,
    WIKI_PASSAGES_URL, WIKI_PASSAGES_FILE,
    DATASETS,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
 

#-- now I'mm gonna use this below
#note that for purposes of speed, most of this file was AI-gen'd
 
def ensure_dirs():
  for d in [DATA_DIR, WIKI_DIR, INDEX_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest: str, chunk_size: int = 1 << 20):
    """Stream-download a file with a progress bar."""
    import urllib.request
    log.info(f"Downloading {url} → {dest}")
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)
            ) as bar:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
        os.rename(tmp, dest)
        log.info(f"Saved to {dest}")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e


