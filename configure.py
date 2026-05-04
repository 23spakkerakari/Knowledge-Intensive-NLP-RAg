"""
The baseline for setting directories, literals, datasets, and initializing params

"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
INDEX_DIR = "./faiss_index"
WIKI_DIR = "./wikipedia"

WIKI_URLS = (
  "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/"
  "psgs_w100.tsv.gz"
)

WIKI_PASSAGES_FILE = os.path.join(WIKI_DIR, "psgs_w100.tsv")

DPR_INDEX_URL  = (
    "https://dl.fbaipublicfiles.com/dpr/checkpoint/retrieval/"
    "single/nq/hnsw_faiss_index.pkl"
)

HF_INDEX_NAME  = "wiki_dpr"                    
HF_INDEX_SPLIT = "train"
 
RAG_MODEL_NAME      = "facebook/rag-sequence-nq"  
RAG_TOKENIZER_NAME  = "facebook/rag-sequence-nq"
DPR_CTX_ENCODER     = "facebook/dpr-ctx_encoder-multiset-base"
DPR_Q_ENCODER       = "facebook/dpr-question_encoder-multiset-base"
BART_MODEL_NAME     = "facebook/bart-large"


DATASETS = {
    "nq":    {"hf_name": "nq_open",       "hf_config": None},
    "trivia": {"hf_name": "trivia_qa",    "hf_config": "unfiltered.nocontext"},
    "wq":    {"hf_name": "web_questions", "hf_config": None},
}
 
'''
Here, we choose hyperparameters that match the outline given in og paper
'''
TRAIN_CONFIG = {
    "learning_rate":        1e-5,
    "weight_decay":         0.01,
    "max_grad_norm":        1.0,
    "num_train_epochs":     3,
    "per_device_train_batch_size": 4,   # subj to change if we have enough vram space
    "gradient_accumulation_steps":  4,  
    "warmup_ratio":         0.06,
    "logging_steps":        50,
    "eval_steps":           500,
    "save_steps":           500,
    "fp16":                 True,
    "dataloader_num_workers": 4,
    "n_docs":               5,          #we set k=5
}
 
GENERATION_CONFIG = {
    "num_beams":         4,
    "max_new_tokens":    32,
    "early_stopping":    True,
    "n_docs":            5,
}
 
# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE    = 8
MAX_ANSWER_LENGTH  = 32 