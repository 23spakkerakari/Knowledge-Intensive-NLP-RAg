import json
import logging 
import os
from typing import Dict, List, Optional
 
import torch
from torch.utils.data import Dataset, DataLoader
import re
import string.punctuation as punc

#normalizing answer
def normalize_answer(s: str):
  #applying from the rag paper
  s = re.sub(r"\b(a|an|the)\b", " ", s)
  s = ''.join(s.split())
  def remove_punc(t):
        exclude = set(punc)
        return "".join(ch for ch in t if ch not in exclude)
    
  s = remove_punc(s)
  return s.lower()

  def exact_match_score(prediction: str, ground_truths: List[str]) -> bool:
    """Return True if the normalised prediction matches any ground truth."""
    pred_norm = normalise_answer(prediction)
    return any(pred_norm == normalise_answer(gt) for gt in ground_truths)
 
 
class QADataset(Dataset):
    """
    Loads a JSONL file produced by download_data.py.
    Each line: {"question": "...", "answers": ["...", ...]}
 
    __getitem__ returns raw strings; the collator handles tokenisation
    so we can change the tokenizer without changing this class.
    """
 
    def __init__(self, jsonl_path: str, max_examples: Optional[int] = None):
        self.examples = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_examples is not None and i >= max_examples:
                    break
                self.examples.append(json.loads(line))
        log.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")
 
    def __len__(self):
        return len(self.examples)
 
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "question": ex["question"],
            "answers":  ex["answers"],        # list[str], multiple valid answers
        }
 
 
 
class RAGCollator:
    """
    Tokenises a batch of {question, answers} dicts for RagSequenceForGeneration.
 
    The HuggingFace RAG model expects:
        input_ids        [B, q_len]
        attention_mask   [B, q_len]
        labels           [B, a_len]   — padded with -100 (ignored in loss)
 
    We use the *first* answer string as the training target, which is standard
    for NQ and TriviaQA.  All answers are kept for evaluation.
    """
 
    def __init__(
        self,
        tokenizer,
        max_question_length: int = 128,
        max_answer_length:   int = 32,
    ):
        self.tokenizer           = tokenizer
        self.max_question_length = max_question_length
        self.max_answer_length   = max_answer_length
 
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        questions = [ex["question"] for ex in batch]
        # Use first answer as training target (standard practice)
        target_answers = [ex["answers"][0] for ex in batch]
        all_answers    = [ex["answers"]    for ex in batch]
 
        # Tokenise questions
        q_enc = self.tokenizer(
            questions,
            max_length=self.max_question_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
 
        # Tokenise answers — these become the decoder labels
        # We use the generator tokenizer (same as question tokenizer for BART)
        with self.tokenizer.as_target_tokenizer():
            a_enc = self.tokenizer(
                target_answers,
                max_length=self.max_answer_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
 
        # Replace padding token id with -100 so it's ignored in loss
        labels = a_enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
 
        return {
            "input_ids":      q_enc["input_ids"],
            "attention_mask": q_enc["attention_mask"],
            "labels":         labels,
            # Keep all answers for EM evaluation (not used in loss)
            "all_answers":    all_answers,
        }
 
 
#here we make the dataloader 
def make_dataloader(
    jsonl_path:          str,
    tokenizer,
    batch_size:          int,
    shuffle:             bool = True,
    max_question_length: int  = 128,
    max_answer_length:   int  = 32,
    num_workers:         int  = 4,
    max_examples:        Optional[int] = None,
) -> DataLoader:
    dataset  = QADataset(jsonl_path, max_examples=max_examples)
    collator = RAGCollator(tokenizer, max_question_length, max_answer_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )


