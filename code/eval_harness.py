import json
import time
from typing import Callable, List
from tqdm import tqdm

from dataset import exact_match_score, squad_f1

def load_dataset(jsonl_path: str, max_examples: int = None):
  examples = []
  with open(jsonl_path) as f:
    for i, line in enumerate(f):
      if max_examples is not None and i>=max_examples:
        break
      examples.append(json.loads(line))
  return examples

def eval_pipeline(retriever, generator, dataset: List[dict], k: int = 5):
  """
  Over here, we go through the following pipeline
  retriever -> gen -> metric on any given dataset
  """

  preds = []
  scores = []
  
  iterator = tqdm(dataset, desc="Evaluating...")
  t0 = time.time()

  for ex in iterator:
    try:
      docs = retriever(ex['question'], k=k)
    except Exception as e:
      import traceback
      print(f"\nRetriever failed on '{ex['question']}'")
      traceback.print_exc()      # ← shows full stack trace
      docs = []

    try:
      pred = generator(ex['question'], docs)
    except Exception as e:
      print(f"Generator failed on '{ex['question']}'")
      traceback.print_exc()
      pred = ""

    em = exact_match_score(pred, ex['answers'])
    f1 = squad_f1(pred, ex['answers'])
    preds.append({
      "question": ex['question'], 
      'prediction': pred, 
      'answers': ex['answers'], 
      'EM_score': em,
      'F1_score': f1,
      'retrieved': [{"title":d.get('title', ''), 'text':d.get('text')[:250]} for d in docs[:5]]
    })

    scores.append({'em':em, 'f1':f1})

  return {
    "em_score": sum(s['em'] for s in scores)/len(scores) if scores else 0.0, 
    'f1_score': sum(s['f1'] for s in scores)/len(scores) if scores else 0.0,
    "n": len(scores), 
    "elapsed": time.time()-t0,
    "predictions": preds
  }


def save_results(result: dict, output_path: str):
    """Save full predictions + scores for later inspection."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def print_summary(result: dict, label: str = ""):
    """Print a short summary of an eval run."""
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"{'='*60}")
    print(f"  EM Score:    {result['em_score']:.2f}")
    print(f"  F1 Score:    {result['f1_score']:.2f}")
    print(f"  Examples: {result['n']:,}")
    print(f"  Time:     {result['elapsed']:.1f}s ({result['elapsed']/result['n']:.2f}s/ex)")
