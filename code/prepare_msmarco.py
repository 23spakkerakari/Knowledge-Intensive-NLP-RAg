import argparse
import json
import os
from datasets import load_dataset


BAD_ANSWERS = {"", "[]", "No Answer Present", "No answer present."}


def is_no_answer(text):
    t = str(text).strip().lower()
    t = t.replace(".", "").replace("[", "").replace("]", "").strip()
    return (
        t == ""
        or t == "no answer present"
        or t == "no answer"
        or t == "none"
    )


def clean_answers(x):
    if x is None:
        return []

    if isinstance(x, str):
        x = x.strip()
        if is_no_answer(x):
            return []
        return [x]

    if isinstance(x, (list, tuple)):
        out = []
        for a in x:
            a = str(a).strip()
            if not is_no_answer(a):
                out.append(a)
        return out

    return []


def convert_split(ds, out_path, max_examples=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            question = ex.get("query") or ex.get("question")
            if not question:
                continue

            # MS-MARCO NLG-style full sentence answers.
            answers = clean_answers(ex.get("wellFormedAnswers"))

            # Fallback if wellFormedAnswers is unavailable/empty.
            if not answers:
                answers = clean_answers(ex.get("answers"))

            if not answers:
                continue

            row = {
                "question": question,
                "answers": answers,
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

            if max_examples is not None and kept >= max_examples:
                break

    print(f"Wrote {kept:,} examples to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train", type=int, default=20000)
    parser.add_argument("--max_dev", type=int, default=1000)
    args = parser.parse_args()

    print("Loading MS-MARCO v2.1 from HuggingFace...")
    ds = load_dataset("ms_marco", "v2.1", trust_remote_code=True)
    print(ds)

    convert_split(ds["train"], "data/msmarco_train.jsonl", max_examples=args.max_train)
    convert_split(ds["validation"], "data/msmarco_dev.jsonl", max_examples=args.max_dev)


if __name__ == "__main__":
    main()
