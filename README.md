# RAG, but with ColBERT and LLaMA

## 1. Introduction

This repo is our re-implementation of **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (Lewis et al., NeurIPS 2020). The paper introduces RAG-Sequence and RAG-Token, which combine a DPR retriever with a BART generator that's jointly fine-tuned end-to-end on open-domain QA. We rebuild that baseline and then swap in two modern alternatives (ColBERT retriever, LLaMA generator) to see how the pieces hold up.

## 2. Chosen Result

We aimed to reproduce the RAG-Sequence Exact Match scores on **Natural Questions** and add a **MS-MARCO** abstractive comparison (Table 1, Lewis et al. 2020, EM = 44.5 on NQ for RAG-Sequence). The NQ result is the headline number of the paper and the cleanest test of whether the retrieve-then-generate setup actually beats a closed-book seq2seq.

## 3. GitHub Contents

```
code/                main retriever/generator modules + notebooks (ColBERT, LLaMA, RAG)
data/               NQ, TriviaQA, WQ, MS-MARCO dev/train JSONL + small_nq_index/
results/            checkpoints, prediction JSONLs, eval JSONs per config
report/             results_summary.md (per-config breakdown)
poster/             final poster
train_msmarco.py    RAG-Sequence training loop (Accelerate + AdamW)
eval_msmarco_metrics.py   ROUGE-L / BLEU-1 / EM / F1 evaluator
prepare_msmarco.py  builds msmarco_train.jsonl / msmarco_dev.jsonl from HF
```

## 4. Re-implementation Details

- **Models:** `facebook/rag-sequence-nq` (DPR + BART), ColBERT-v2 retriever, `facebook/bart-large` generator fine-tuned on 5k ColBERT contexts, and a zero-shot LLaMA-Instruct generator.
- **Datasets:** Natural Questions (open), TriviaQA, MS-MARCO v2.1 (well-formed answers).
- **Tools:** HuggingFace Transformers, Accelerate, FAISS, ColBERT-v2, PyTorch, fp16.
- **Metrics:** EM and SQuAD-F1 for NQ; EM, F1, ROUGE-L, BLEU-1 for abstractive MS-MARCO.
- **Modifications:** swapped HF's RagRetriever for a custom FAISS index over a 50k passage subset (`data/small_nq_index/`), and added an MS-MARCO answerable-only filter after the full-mix run collapsed onto `"No Answer Present."` predictions.

## 5. Reproduction Steps

```bash
# 1. install
pip install torch transformers accelerate datasets faiss-cpu colbert-ai

# 2. build datasets (NQ/Trivia/WQ are downloaded by code/download_data.py)
python prepare_msmarco.py --max_train 20000 --max_dev 1000

# 3. train RAG-Sequence on a dataset
python train_msmarco.py --dataset msmarco --epochs 1 --batch_size 4 --fp16

# 4. evaluate a checkpoint
python eval_msmarco_metrics.py \
  --model_dir results/rag_baseline_msmarco_50k_answerable/checkpoint-1000 \
  --data_path data/msmarco_dev.jsonl --max_examples 500 --num_beams 4

# 5. ColBERT + BART / LLaMA pipelines are in code/main_colbert*.ipynb
```

Compute: trained on a single Colab T4/A100 (16 GB) in fp16. Each MS-MARCO 50k run takes ~1.5 hours; eval on 500 dev examples is ~1-7 minutes depending on the generator.

## 6. Results / Insights

| Config                                 | Dataset / N        | EM        | F1        | Notes                        |
| -------------------------------------- | ------------------ | --------- | --------- | ---------------------------- |
| Lewis et al. 2020 (original)           | NQ test            | 44.5      | n/a       | full Wikipedia, full train   |
| ColBERT + vanilla BART                 | NQ / 3610          | 0.00      | 2.08      | negative control             |
| **ColBERT + BART finetuned (5k)**      | **NQ / 500**       | **21.60** | **27.26** | best NQ on my setup          |
| ColBERT + LLaMA (zero-shot)            | NQ / 500           | 2.60      | 8.48      | hedges instead of answering  |
| RAG-Seq on MS-MARCO 50k (full)         | MS-MARCO / 500     | 0.40      | 1.45      | collapses to "No Answer" 97% |
| **RAG-Seq on MS-MARCO 50k answerable** | **MS-MARCO / 500** | **2.80**  | **33.42** | best MS-MARCO on my setup    |
| ColBERT + LLaMA (zero-shot)            | MS-MARCO / 500     | 0.20      | 19.88     | fluent but paraphrased       |

We don't hit the 44.5 EM number from the paper (we used a 50k-passage subset instead of full Wikipedia and a smaller train set), but the qualitative ordering matches: jointly fine-tuned RAG/BART beats vanilla seq2seq by a huge margin, and filtering bad training labels matters more than adding epochs.

## 7. Conclusion

Two things mattered way more than we expected: (1) fine-tuning the generator on retriever-shaped contexts is what unlocks open-domain QA (vanilla BART scored 0% EM even when the retriever literally fed it the answer string), and (2) the MS-MARCO `"No Answer Present"` rows poison training if you don't filter them out. Zero-shot LLaMA is fluent on MS-MARCO but won't compete with a fine-tuned RAG on short-answer NQ.

## 8. References

- Lewis et al. 2020. _Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks._ NeurIPS 2020. https://arxiv.org/abs/2005.11401
- Karpukhin et al. 2020. _Dense Passage Retrieval for Open-Domain Question Answering._ EMNLP 2020.
- Khattab & Zaharia. 2020. _ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT._ SIGIR 2020.
- Touvron et al. 2023. _LLaMA: Open and Efficient Foundation Language Models._
- HuggingFace Transformers + Accelerate; FAISS; MS-MARCO v2.1; Natural Questions; TriviaQA.

## 9. Acknowledgements

Done as the final project for **CS 4782 (Deep Learning), Cornell University, Spring 2026**. Huge thanks to the course staff for the project framing and feedback, and to the authors of the RAG / DPR / ColBERT codebases that this builds on.

And of course:

We 🫶 Kilian
