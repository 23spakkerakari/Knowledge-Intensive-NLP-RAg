# Results Summary

Quick walk through everything in `results/`, grouped by train/eval config. I'm using EM and SQuAD-F1 as the main numbers (plus ROUGE-L and BLEU-1 where the gold answers are full sentences, i.e. MS-MARCO). Scores from the JSON dumps are stored as fractions; I converted them to percent.

## 1. ColBERT + LLaMA (zero-shot)

Files: `colbert_llama_nq_predictions.jsonl`, `colbert_llama_msmarco_predictions.jsonl` (500 lines each).

| Dataset | N | EM | F1 | ROUGE-L | BLEU-1 |
|---|---|---|---|---|---|
| NQ dev | 500 | 2.60 | 8.48 | 9.31 | 3.46 |
| MS-MARCO dev | 500 | 0.20 | 19.88 | 17.90 | 16.25 |

Sample: gold = `Bobby Scott / Bob Russell`, pred = `The song "He Ain't Heavy, He's My Brother" was written by Bobby Scott and Bob Russell.` LLaMA paraphrases instead of giving the short span, so F1 holds up but EM tanks.

## 2. ColBERT + vanilla BART, full NQ dev

Files: `colbert_open_domain_eval/{run_config.json, results_colbert_bart_k5.json}`. Full NQ dev (3,610 examples), k=5.

| EM | F1 | N |
|---|---|---|
| 0.00 | 2.08 | 3610 |

Sample: `when was the last time anyone was on the moon` -> `The last time anyone was on the moon was Apollo 11. It was launched on November 14, 1969...`. Off-the-shelf BART is just hallucinating.

## 3. ColBERT + vanilla BART, small NQ index, 500 slice

Files: `colbert_open_domain_eval_small_nq_corpus/{run_config.json, results_colbert_bart_small_nq_k5_500.json}`.

| EM | F1 | N |
|---|---|---|
| 0.20 | 7.28 | 500 |

Sample: retriever literally gave it `Question: how many seasons of the bastard executioner are there Answer: one` as doc 1, and vanilla BART still predicts `There are 7 seasons of the bastard executioner.`

## 4. ColBERT + BART fine-tuned on 5k ColBERT contexts (BEST on NQ)

Files: `colbert_open_domain_eval_finetuned_bart/{run_config.json, results_colbert_bart_small_nq_k5_500.json}`.

| EM | F1 | N |
|---|---|---|
| 21.60 | 27.26 | 500 |

Same retriever and slice as #3, only the generator changed. Sample: gold `December 1972`, pred `11 December 1972` (F1 ~0.8). This is the headline NQ number.

## 5. BART adaptation artifact (training data + config)

Files: `colbert_bart_adapted_generator/nq_train_colbert_contexts_5k.jsonl` (5,000 lines) and `colbert_bart_adapted_generator/bart_colbert_nq_5k/{config.json, generation_config.json}`. No weights in this snapshot. This is the source-of-truth training data for the #4 generator (question + top-5 ColBERT passages -> short gold answer).

## 6. Adapted-BART eval slots (empty)

Files: `colbert_open_domain_eval_adapted_bart_5k/`, `colbert_open_domain_eval_adapted_bart_5k_k50/`. Both are empty directories. Placeholders for k=5 and k=50 sweeps that never ran.

## 7. RAG-Sequence on NQ

Files: `rag_baseline_nq/{best, checkpoint-500..16000}/` (config + tokenizers only, no weights or JSONL eval here) plus 14 tfevents files in `rag_baseline_nq/rag_baseline/` (only one is meaningfully sized at 32 KB).

No standalone prediction dump on disk, so no clean EM/F1 to report from this snapshot.

## 8. RAG-Sequence on TriviaQA

Files: `trivia_run1/rag_baseline/events.out.tfevents...` (single 88-byte stub). Run never produced a checkpoint or eval. No metrics.

## 9. RAG-Sequence on MS-MARCO 50k (full mix)

Files: `rag_baseline_msmarco_50k/{checkpoint-500, checkpoint-1000}/` (config + tokenizers only) and `msmarco_50k_predictions.jsonl` (500 lines).

| EM | F1 | ROUGE-L | BLEU-1 | N | "No Answer" preds |
|---|---|---|---|---|---|
| 0.40 | 1.45 | 1.46 | 0.04 | 500 | 487 (97.4%) |

The model basically just outputs `"No Answer Present."` on almost every dev question.

## 10. RAG-Sequence on MS-MARCO 50k answerable, 1 epoch (BEST on MS-MARCO)

Files: `rag_baseline_msmarco_50k_answerable/checkpoint-1000/` and `msmarco_50k_answerable_predictions.jsonl` (500 lines).

| EM | F1 | ROUGE-L | BLEU-1 | N | "No Answer" preds |
|---|---|---|---|---|---|
| 2.80 | 33.42 | 32.14 | 20.10 | 500 | 0 (0%) |

Samples: `can you burn your lawn with fertilizer` -> `Yes, you can burn your lawn with fertilizer.` (near-perfect). One residual issue is decoder repetition on some examples (`abdominal pain, abdominal pain, abdominal pain...`).

## 11. RAG-Sequence on MS-MARCO 50k answerable, 3 epochs

Files: `rag_baseline_msmarco_50k_answerable_3ep/{checkpoint-1000, checkpoint-2000, checkpoint-3000}/`. No `_3ep` prediction JSONL on disk, so 3-epoch numbers are unmeasured here.

## Cross-config table

| # | Config | Dataset / N | EM | F1 |
|---|---|---|---|---|
| 1 | ColBERT + LLaMA (zero-shot) | NQ / 500 | 2.60 | 8.48 |
| 1 | ColBERT + LLaMA (zero-shot) | MS-MARCO / 500 | 0.20 | 19.88 |
| 2 | ColBERT + vanilla BART | NQ / 3610 | 0.00 | 2.08 |
| 3 | ColBERT + vanilla BART, small NQ idx | NQ / 500 | 0.20 | 7.28 |
| 4 | **ColBERT + BART finetuned on 5k** | **NQ / 500** | **21.60** | **27.26** |
| 6 | adapted-BART k=5 / k=50 | NQ / 500 | not run | not run |
| 7 | RAG-Seq on NQ | NQ | no eval dump | no eval dump |
| 8 | RAG-Seq on TriviaQA | trivia | no run | no run |
| 9 | RAG-Seq on MS-MARCO 50k (full) | MS-MARCO / 500 | 0.40 | 1.45 |
| 10 | **RAG-Seq on MS-MARCO 50k answerable, 1 ep** | **MS-MARCO / 500** | **2.80** | **33.42** |
| 11 | RAG-Seq on MS-MARCO 50k answerable, 3 ep | MS-MARCO / 500 | no eval dump | no eval dump |
