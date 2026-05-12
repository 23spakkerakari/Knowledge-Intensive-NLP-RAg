
# import os
# import gzip
# import json
# import argparse
# import logging
# from pathlib import Path
# from tqdm import tqdm
 
# import datasets as hf_datasets
 
# from configure import (
#     DATA_DIR, WIKI_DIR, INDEX_DIR,
#     WIKI_PASSAGES_URL, WIKI_PASSAGES_FILE,
#     DATASETS,
# )


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )
# log = logging.getLogger(__name__)
 

# #-- now I'mm gonna use this below
# #note that for purposes of speed, most of this file was AI-gen'd
 
# def ensure_dirs():
#   for d in [DATA_DIR, WIKI_DIR, INDEX_DIR]:
#     Path(d).mkdir(parents=True, exist_ok=True)

# def download_file(url: str, dest: str, chunk_size: int = 1 << 20):
#     """Stream-download a file with a progress bar."""
#     import urllib.request
#     log.info(f"Downloading {url} → {dest}")
#     tmp = dest + ".part"
#     try:
#         with urllib.request.urlopen(url) as resp:
#             total = int(resp.headers.get("Content-Length", 0))
#             with open(tmp, "wb") as f, tqdm(
#                 total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)
#             ) as bar:
#                 while True:
#                     chunk = resp.read(chunk_size)
#                     if not chunk:
#                         break
#                     f.write(chunk)
#                     bar.update(len(chunk))
#         os.rename(tmp, dest)
#         log.info(f"Saved to {dest}")
#     except Exception as e:
#         if os.path.exists(tmp):
#             os.remove(tmp)
#         raise e

import os
import gzip
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
 
import datasets as hf_datasets
 
from configure import (
    DATA_DIR,
    DATASETS,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
 



# ── Per-dataset answer extraction ─────────────────────────────────────────────
# Each HF dataset stores answers differently, so we normalise them here into
# a flat list of strings — the format QADataset expects.

def extract_answers(example: dict, dataset_key: str):
    if dataset_key == "nq":
        # nq_open: "answer" is already a list of strings
        return example["answer"]
    elif dataset_key == "trivia":
        # trivia_qa: answers live in example["answer"]["aliases"]
        return example["answer"]["aliases"]
    elif dataset_key == "wq":
        # web_questions: "answers" is a list of strings
        return example["answers"]
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")


# ── HF split name → our file suffix ───────────────────────────────────────────
# Some datasets don't have all three splits, so we only write what exists.

SPLIT_MAP = {
    "train":      "train",
    "validation": "dev",
    "test":       "test",
}


def download_dataset(dataset_key: str, max_examples: int = None):
    """
    Download one QA dataset from HuggingFace and write it to JSONL files
    under DATA_DIR.  Each line: {"question": "...", "answers": ["...", ...]}
    """
    cfg = DATASETS[dataset_key]
    hf_name   = cfg["hf_name"]
    hf_config = cfg["hf_config"]

    log.info(f"Loading '{hf_name}' (config={hf_config}) from HuggingFace ...")
    ds = hf_datasets.load_dataset(hf_name, hf_config, trust_remote_code=True)

    for hf_split, file_suffix in SPLIT_MAP.items():
        if hf_split not in ds:
            log.info(f"  '{dataset_key}' has no '{hf_split}' split - skipping.")
            continue

        split_data = ds[hf_split]
        if max_examples is not None:
            split_data = split_data.select(range(min(max_examples, len(split_data))))

        out_path = os.path.join(DATA_DIR, f"{dataset_key}_{file_suffix}.jsonl")
        log.info(f"  Writing {len(split_data):,} examples to {out_path}")

        with open(out_path, "w", encoding="utf-8") as fh:
            for example in tqdm(split_data, desc=f"{dataset_key}/{file_suffix}"):
                record = {
                    "question": example["question"],
                    "answers":  extract_answers(example, dataset_key),
                }
                fh.write(json.dumps(record) + "\n")

        log.info(f"  Saved {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download NQ, TriviaQA, and WebQuestions and save as JSONL."
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Which datasets to download.",
    )
    p.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap each split (useful for quick testing).",
    )
    return p.parse_args()


def ensure_dirs():
    for d in [DATA_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    ensure_dirs()
    for key in args.datasets:
        log.info(f"=== Downloading {key} ===")
        download_dataset(key, max_examples=args.max_examples)
    log.info("All done. Data saved to ./data/")


