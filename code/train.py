import argparse
import logging
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    RagSequenceForGeneration,
    RagRetriever,
    RagTokenizer,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed

from config import (
    DATA_DIR, OUTPUT_DIR,
    RAG_MODEL_NAME, RAG_TOKENIZER_NAME, INDEX_NAME,
    TRAIN_CONFIG, GENERATION_CONFIG,
)
from dataset import make_dataloader, exact_match_score
from checkpoint_utils import (
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    get_run_state,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="nq", choices=["nq", "trivia", "wq"])
    p.add_argument("--output_dir", default=None)
    p.add_argument("--n_docs",     type=int, default=TRAIN_CONFIG["n_docs"])
    p.add_argument("--epochs",     type=int, default=TRAIN_CONFIG["num_train_epochs"])
    p.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["per_device_train_batch_size"])
    p.add_argument("--grad_accum", type=int, default=TRAIN_CONFIG["gradient_accumulation_steps"])
    p.add_argument("--lr",         type=float, default=TRAIN_CONFIG["learning_rate"])
    p.add_argument("--fp16",       action="store_true", default=TRAIN_CONFIG["fp16"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--no_resume",  action="store_true",
                   help="Start training from scratch even if checkpoints exist")
    p.add_argument("--eval_only",  action="store_true",
                   help="Skip training; just evaluate the latest checkpoint")
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=TRAIN_CONFIG["eval_steps"])
    p.add_argument("--save_steps", type=int, default=TRAIN_CONFIG["save_steps"])
    p.add_argument("--logging_steps", type=int, default=TRAIN_CONFIG["logging_steps"])
    p.add_argument("--keep_last_n_ckpts", type=int, default=2,
                   help="How many old checkpoints to keep on Drive")
    return p.parse_args()

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_path: str):
    """Load model with our custom FAISS-based retriever (no load_dataset)."""
    import sys
    sys.path.insert(0, "/content/rag_baseline")
    from custom_retriever import CustomRagRetriever
    from transformers import RagConfig
    
    log.info(f"Loading tokenizer: {model_path}")
    tokenizer = RagTokenizer.from_pretrained(model_path)
    
    log.info("Loading config...")
    config = RagConfig.from_pretrained(model_path)
    
    log.info("Loading custom retriever (pre-built FAISS index)...")
    retriever = CustomRagRetriever(
        config=config,
        question_encoder_tokenizer=tokenizer.question_encoder,
        generator_tokenizer=tokenizer.generator,
        index_path="/content/wiki_dpr_built/wiki_dpr_nq.faiss",
        passages_path="/content/wiki_dpr_built/passages.parquet",
    )
    
    log.info(f"Loading model: {model_path}")
    model = RagSequenceForGeneration.from_pretrained(
        model_path, retriever=retriever
    )
    
    # Freeze the document encoder (paper convention)
    if hasattr(model.rag, "context_encoder_training"):
        for param in model.rag.context_encoder_training.parameters():
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable parameters: {trainable:,} / {total:,}")
    return model, tokenizer

# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, eval_loader, accelerator, n_docs, max_new_tokens=32, num_beams=1):
    model.eval()
    correct, total = 0, 0
    for batch in eval_loader:
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            num_beams=num_beams,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            n_docs=n_docs,
        )
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for pred, answers in zip(preds, batch["all_answers"]):
            if exact_match_score(pred, answers):
                correct += 1
            total += 1

    correct_t = torch.tensor(correct, device=accelerator.device)
    total_t   = torch.tensor(total,   device=accelerator.device)
    correct_t = accelerator.reduce(correct_t, reduction="sum")
    total_t   = accelerator.reduce(total_t,   reduction="sum")

    em = (correct_t / total_t).item() * 100
    model.train()
    return {"em": em, "correct": int(correct_t), "total": int(total_t)}

# ── Save helpers ──────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, output_dir, step):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    log.info(f"Saved checkpoint → {ckpt_dir}")

def save_best(model, tokenizer, output_dir):
    best_dir = os.path.join(output_dir, "best")
    Path(best_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    log.info(f"Saved best → {best_dir}")

# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(OUTPUT_DIR, f"rag_baseline_{args.dataset}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    state = get_run_state(args.output_dir)
    log.info(f"Run state: {state}")

    resume_from = None
    if not args.no_resume:
        resume_from = find_latest_checkpoint(args.output_dir)
        if resume_from:
            log.info(f"Auto-resuming from: {resume_from}")
        else:
            log.info("No checkpoint found — starting from pre-trained RAG.")

    model_path = resume_from if resume_from else RAG_MODEL_NAME

    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16 else "no",
        gradient_accumulation_steps=args.grad_accum,
    )

    model, tokenizer = load_model_and_tokenizer(model_path)

    train_path = os.path.join(DATA_DIR, f"{args.dataset}_train.jsonl")
    dev_path   = os.path.join(DATA_DIR, f"{args.dataset}_dev.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}. Run colab_setup.py first.")

    eval_loader = make_dataloader(
        dev_path, tokenizer, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=2,
    )

    if args.eval_only:
        eval_loader_p = accelerator.prepare(eval_loader)
        model = accelerator.prepare(model)
        log.info("Running eval-only with beam search …")
        m = evaluate(
            accelerator.unwrap_model(model), tokenizer, eval_loader_p, accelerator,
            n_docs=args.n_docs, num_beams=4,
        )
        log.info(f"Final EM = {m['em']:.2f}% ({m['correct']}/{m['total']})")
        return

    train_loader = make_dataloader(
        train_path, tokenizer, batch_size=args.batch_size, shuffle=True,
        num_workers=2, max_examples=args.max_train_examples,
    )

    q_enc_params = list(model.rag.question_encoder.parameters())
    gen_params   = list(model.rag.generator.parameters())

    optimizer = AdamW(
        [
            {"params": q_enc_params, "lr": args.lr * 0.1},
            {"params": gen_params,   "lr": args.lr},
        ],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * TRAIN_CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler,
    )

    global_step = state["latest_step"] if resume_from else 0
    best_em     = 0.0
    log.info(f"Starting at global_step = {global_step}")

    log.info(f"Training: {args.epochs} epochs, {total_steps} total steps")

    for epoch in range(args.epochs):
        model.train()
        running_loss, num_batches = 0.0, 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    n_docs=args.n_docs,
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["max_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            num_batches  += 1

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / num_batches
                    lr_now   = scheduler.get_last_lr()[0]
                    log.info(f"Epoch {epoch+1} | Step {global_step:>6} | Loss {avg_loss:.4f} | LR {lr_now:.2e}")
                    running_loss, num_batches = 0.0, 0

                if global_step % args.eval_steps == 0:
                    m = evaluate(
                        accelerator.unwrap_model(model), tokenizer,
                        eval_loader, accelerator, n_docs=args.n_docs, num_beams=1,
                    )
                    log.info(f"[Eval] Step {global_step} | EM = {m['em']:.2f}% ({m['correct']}/{m['total']})")
                    if m["em"] > best_em:
                        best_em = m["em"]
                        if accelerator.is_main_process:
                            save_best(accelerator.unwrap_model(model), tokenizer, args.output_dir)
                        log.info(f"  New best EM: {best_em:.2f}%")

                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    save_checkpoint(accelerator.unwrap_model(model), tokenizer, args.output_dir, global_step)
                    cleanup_old_checkpoints(args.output_dir, keep_last_n=args.keep_last_n_ckpts)

        log.info(f"Epoch {epoch+1} complete.")

    log.info("Final evaluation (num_beams=4) …")
    m = evaluate(
        accelerator.unwrap_model(model), tokenizer, eval_loader, accelerator,
        n_docs=args.n_docs, num_beams=4,
    )
    log.info(f"[Final] EM = {m['em']:.2f}% ({m['correct']}/{m['total']})")
    log.info(f"Best checkpoint: {os.path.join(args.output_dir, 'best')}")

if __name__ == "__main__":
    args = parse_args()
    train(args)
