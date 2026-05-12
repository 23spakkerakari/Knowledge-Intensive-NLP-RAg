import argparse
import json
import math
import os
import re
from collections import Counter

import torch
from tqdm import tqdm

from train_msmarco import load_model_and_tokenizer


def normalize_tokens(text):
    return re.findall(r"\w+", str(text).lower())


def lcs_len(a, b):
    """Longest common subsequence length for ROUGE-L."""
    dp = [0] * (len(b) + 1)

    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            temp = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[-1]


def rouge_l_f1(pred, ref):
    pred_toks = normalize_tokens(pred)
    ref_toks = normalize_tokens(ref)

    if not pred_toks or not ref_toks:
        return 0.0

    lcs = lcs_len(pred_toks, ref_toks)
    precision = lcs / len(pred_toks)
    recall = lcs / len(ref_toks)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def corpus_bleu1(preds, refs_list):
    """Corpus BLEU-1 with clipped unigram precision and brevity penalty."""
    clipped_total = 0
    pred_total = 0
    pred_len_total = 0
    ref_len_total = 0

    for pred, refs in zip(preds, refs_list):
        pred_toks = normalize_tokens(pred)
        ref_tokens_list = [normalize_tokens(r) for r in refs if str(r).strip()]

        if not pred_toks or not ref_tokens_list:
            continue

        pred_counts = Counter(pred_toks)

        best_ref_counts = Counter()
        best_overlap = -1
        best_ref_len = len(ref_tokens_list[0])

        for ref_toks in ref_tokens_list:
            ref_counts = Counter(ref_toks)
            overlap = sum((pred_counts & ref_counts).values())
            if overlap > best_overlap:
                best_overlap = overlap
                best_ref_counts = ref_counts
                best_ref_len = len(ref_toks)

        clipped_total += sum((pred_counts & best_ref_counts).values())
        pred_total += len(pred_toks)
        pred_len_total += len(pred_toks)
        ref_len_total += best_ref_len

    if pred_total == 0:
        return 0.0

    precision = clipped_total / pred_total

    if pred_len_total == 0:
        bp = 0.0
    elif pred_len_total > ref_len_total:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len_total / pred_len_total)

    return 100 * bp * precision


def load_jsonl(path, max_examples=None):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if max_examples is not None and len(examples) >= max_examples:
                break

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", default="data/msmarco_dev.jsonl")
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_docs", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--out_path", default="outputs/msmarco_predictions.jsonl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model:", args.model_dir)
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    model.to(device)
    model.eval()

    examples = load_jsonl(args.data_path, max_examples=args.max_examples)

    preds = []
    refs_list = []

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as out_f:
        for start in tqdm(range(0, len(examples), args.batch_size)):
            batch = examples[start:start + args.batch_size]
            questions = [ex["question"] for ex in batch]

            enc = tokenizer.question_encoder(
                questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                generated = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    n_docs=args.n_docs,
                    num_beams=args.num_beams,
                    num_return_sequences=1,
                    max_new_tokens=args.max_new_tokens,
                )

            batch_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for ex, pred in zip(batch, batch_preds):
                refs = ex["answers"]

                preds.append(pred)
                refs_list.append(refs)

                out_f.write(json.dumps({
                    "question": ex["question"],
                    "prediction": pred,
                    "references": refs,
                }, ensure_ascii=False) + "\n")

    rouge_l_scores = []
    for pred, refs in zip(preds, refs_list):
        best = max(rouge_l_f1(pred, ref) for ref in refs)
        rouge_l_scores.append(best)

    rouge_l = 100 * sum(rouge_l_scores) / len(rouge_l_scores)
    bleu1 = corpus_bleu1(preds, refs_list)

    print("=" * 80)
    print(f"Examples evaluated: {len(preds)}")
    print(f"ROUGE-L: {rouge_l:.2f}")
    print(f"BLEU-1:  {bleu1:.2f}")
    print(f"Predictions saved to: {args.out_path}")


if __name__ == "__main__":
    main()
